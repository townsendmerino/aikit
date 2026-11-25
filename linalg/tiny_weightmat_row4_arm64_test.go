package linalg

// twRow4F16 is the row4 layout of a canonical int4 weight with binary16 scales, built the
// way RepackInt4Row4 builds it, or nils when this core or shape cannot run it.
func twRow4F16(q4 []byte, q4s16 []uint16, s twShape) ([]byte, []uint16, bool) {
	if !int4Row4Usable(s.rows, s.cols, s.group) {
		return nil, nil, false
	}
	return RepackW4A8Row4(q4, s.rows, s.cols, s.group), RepackW4A8Row4ScalesF16(q4s16, s.rows, s.cols, s.group), true
}
