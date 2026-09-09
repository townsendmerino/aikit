package vision

import (
	"bytes"
	"fmt"
	"image"
	"math"
)

// Gemma 4 vision preprocessing: image bytes → pre-unfolded patches + per-patch
// (x,y) grid position ids, following the real image_processing_gemma4.py
// pipeline. Deliberately NOT sharing Preprocess/Config (preprocess.go) — that
// path is a fixed SQUARE resize with mean/std normalize for SigLIP; Gemma 4
// resizes to an ASPECT-RATIO-PRESERVING, variable (multiple-of-48) target size
// with NO mean/std normalize (image_mean=[0,0,0], image_std=[1,1,1] — a plain
// [0,1] rescale; the [-1,1] remap happens model-side, in Gemma4Encoder.Forward).
//
// gemma4SupportedSoftTokens are the only legal max_soft_tokens budgets
// (image_processing_gemma4.py's _SUPPORTED_SOFT_TOKENS) — the caller picks one;
// the real per-image soft-token count is data-dependent and can be anything
// ≤ the budget (numPatches/9 after resize).
var gemma4SupportedSoftTokens = map[int]bool{70: true, 140: true, 280: true, 560: true, 1120: true}

// Gemma4AspectRatioSize computes the resize target (a multiple of
// poolingKernelSize*patchSize on each axis, aspect-ratio preserved by a single
// uniform scale factor, each axis independently rounded DOWN — never up) for a
// source image of the given height/width, bounded by maxSoftTokens patches worth
// of pixels. Mirrors get_aspect_ratio_preserving_size (image_processing_gemma4.py).
func Gemma4AspectRatioSize(height, width, maxSoftTokens, patchSize, poolingKernelSize int) (targetH, targetW int) {
	sideMult := poolingKernelSize * patchSize
	maxPatches := maxSoftTokens * poolingKernelSize * poolingKernelSize
	targetPx := float64(maxPatches) * float64(patchSize*patchSize)
	totalPx := float64(height) * float64(width)
	factor := math.Sqrt(targetPx / totalPx)
	idealH, idealW := factor*float64(height), factor*float64(width)
	targetH = int(idealH/float64(sideMult)) * sideMult
	targetW = int(idealW/float64(sideMult)) * sideMult
	// Degenerate-dimension clamp (extreme aspect ratios rounding one axis to 0):
	// floor to one side_mult unit and cap the other so the total patch budget
	// isn't exceeded.
	maxSide := (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult
	if targetH <= 0 {
		targetH = sideMult
	}
	if targetW <= 0 {
		targetW = sideMult
	}
	if targetH > maxSide {
		targetH = (maxSide / sideMult) * sideMult
	}
	if targetW > maxSide {
		targetW = (maxSide / sideMult) * sideMult
	}
	return targetH, targetW
}

// Gemma4Preprocess decodes image bytes and returns pre-unfolded patches
// [numPatches, 3*patchSize*patchSize] ([0,1]-scaled, no mean/std) plus per-patch
// (x,y) grid position ids, both in row-major (row,col) patch order — the input
// Gemma4Encoder.Forward expects. maxSoftTokens must be one of the five legal
// budgets {70,140,280,560,1120} (0 defaults to HF's own default, 280).
func Gemma4Preprocess(data []byte, maxSoftTokens int) ([]float32, [][2]int, error) {
	const patchSize = 16
	const poolingKernelSize = 3
	const maxPixels = 16 << 20 // decompression-bomb guard, same bound as Gemma3()
	if maxSoftTokens == 0 {
		maxSoftTokens = 280
	}
	if !gemma4SupportedSoftTokens[maxSoftTokens] {
		return nil, nil, fmt.Errorf("vision: max_soft_tokens %d not one of {70,140,280,560,1120}", maxSoftTokens)
	}
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("vision: decode header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 {
		return nil, nil, fmt.Errorf("vision: non-positive image dims %dx%d", ic.Width, ic.Height)
	}
	if int64(ic.Width)*int64(ic.Height) > maxPixels {
		return nil, nil, fmt.Errorf("vision: image %dx%d exceeds %d-pixel limit (decompression bomb?)", ic.Width, ic.Height, maxPixels)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("vision: decode: %w", err)
	}
	targetH, targetW := Gemma4AspectRatioSize(ic.Height, ic.Width, maxSoftTokens, patchSize, poolingKernelSize)
	if targetH < patchSize || targetW < patchSize {
		return nil, nil, fmt.Errorf("vision: resize target %dx%d smaller than one patch (%d)", targetH, targetW, patchSize)
	}
	hwc := gemma4ResizeToHWC01(img, targetH, targetW)
	patches, positionIDs := gemma4Unfold(hwc, targetH, targetW, patchSize)
	return patches, positionIDs, nil
}

// gemma4ResizeToHWC01 bilinearly resizes img to targetH×targetW (independent
// axes — NOT the square-only newXMap/yTap callers in preprocess.go, though it
// reuses those same per-axis helpers) and writes [0,1]-scaled HWC floats (no
// mean/std subtraction — Gemma 4's image_mean=[0,0,0]/image_std=[1,1,1]).
func gemma4ResizeToHWC01(img image.Image, targetH, targetW int) []float32 {
	nr := toNRGBA(img)
	sw, sh := nr.Rect.Dx(), nr.Rect.Dy()
	stride := nr.Stride
	out := make([]float32, targetH*targetW*3)
	m := newXMap(targetW, sw)
	for dy := range targetH {
		y0, y1, fy := yTap(dy, targetH, sh)
		for dx := range targetW {
			x0, x1, fx := m.x0[dx], m.x1[dx], m.fx[dx]
			for c := range 3 {
				p00 := float64(nr.Pix[y0*stride+x0*4+c])
				p01 := float64(nr.Pix[y0*stride+x1*4+c])
				p10 := float64(nr.Pix[y1*stride+x0*4+c])
				p11 := float64(nr.Pix[y1*stride+x1*4+c])
				top := p00 + (p01-p00)*fx
				bot := p10 + (p11-p10)*fx
				v := float32((top + (bot-top)*fy) / 255.0)
				out[(dy*targetW+dx)*3+c] = v
			}
		}
	}
	return out
}

// gemma4Unfold reshapes an HWC [0,1] image into per-patch flat vectors, matching
// HF's convert_image_to_patches exactly: image.reshape(C,H/p,p,W/p,p)
// .permute(1,3,2,4,0).reshape(-1,p*p*C) — i.e. within each patch the flat order
// is (row-within-patch, col-within-patch, channel), NOT channel-major. Patches
// themselves are enumerated row-major over the (gridH,gridW) grid, with position
// ids (x=col,y=row) — the same row-major convention Gemma4Encoder's averagePool
// bucket formula assumes.
func gemma4Unfold(hwc []float32, targetH, targetW, patchSize int) ([]float32, [][2]int) {
	gridH, gridW := targetH/patchSize, targetW/patchSize
	n := gridH * gridW
	patchDim := patchSize * patchSize * 3
	patches := make([]float32, n*patchDim)
	positionIDs := make([][2]int, n)
	for py := range gridH {
		for px := range gridW {
			idx := py*gridW + px
			positionIDs[idx] = [2]int{px, py}
			dst := patches[idx*patchDim : (idx+1)*patchDim]
			k := 0
			for ry := range patchSize {
				srcY := py*patchSize + ry
				rowBase := srcY * targetW * 3
				for rx := range patchSize {
					srcBase := rowBase + (px*patchSize+rx)*3
					dst[k], dst[k+1], dst[k+2] = hwc[srcBase], hwc[srcBase+1], hwc[srcBase+2]
					k += 3
				}
			}
		}
	}
	return patches, positionIDs
}
