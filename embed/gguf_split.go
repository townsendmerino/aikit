package embed

import (
	"fmt"
	"path/filepath"
	"runtime"

	"github.com/townsendmerino/aikit/mmap"
)

// OpenGGUFSplitMmap opens a llama.cpp split GGUF as one GGUFFile. llama-gguf-split writes a model as
// "<name>-00001-of-0000N.gguf" … each shard a complete GGUF holding some of the tensors, with split.no, split.count
// and (on the first) split.tensors.count in its metadata. Every shard is memory-mapped and parsed; the first shard's
// metadata is the model's, and the tensor directory is the union of the shards'. Tensor, RowDequantizer and Q4KRaw
// then read a tensor from whichever shard holds it, exactly as from a single file.
//
// paths must be the whole set, in shard order. Each shard's split.no and split.count are checked against its place in
// paths, a tensor named by two shards is refused, and the union must match split.tensors.count when the first shard
// records it, so a missing, duplicated or misordered shard fails here rather than as a wrong weight later. One path is
// OpenGGUFMmap. Close unmaps every shard.
func OpenGGUFSplitMmap(paths []string) (*GGUFFile, error) {
	if len(paths) == 0 {
		return nil, errFormatf("gguf: a split set needs at least one shard")
	}
	if len(paths) == 1 {
		return OpenGGUFMmap(paths[0])
	}
	var maps [][]byte
	fail := func(err error) (*GGUFFile, error) {
		for _, m := range maps {
			_ = mmap.Unmap(m)
		}
		return nil, err
	}
	var out *GGUFFile
	for i, p := range paths {
		data, err := mmap.MapReadOnly(p)
		if err != nil {
			return fail(fmt.Errorf("gguf: %w", err))
		}
		maps = append(maps, data)
		s, err := parseGGUF(data)
		if err != nil {
			return fail(fmt.Errorf("gguf: shard %s: %w", filepath.Base(p), err))
		}
		no, okNo := s.Uint("split.no")
		count, okCount := s.Uint("split.count")
		if !okNo || !okCount || no != uint64(i) || count != uint64(len(paths)) {
			return fail(errFormatf("gguf: %s is not shard %d of %d (its split.no %d, split.count %d)", filepath.Base(p), i+1, len(paths), no, count))
		}
		if out == nil {
			out = &GGUFFile{Metadata: s.Metadata, tensors: make(map[string]ggufTensorInfo, len(s.tensors))}
		}
		for name, info := range s.tensors {
			if _, dup := out.tensors[name]; dup {
				return fail(errFormatf("gguf: tensor %q is in more than one shard (again in %s)", name, filepath.Base(p)))
			}
			info.sec = s.data
			out.tensors[name] = info
		}
	}
	if want, ok := out.Uint("split.tensors.count"); ok && want != uint64(len(out.tensors)) {
		return fail(errFormatf("gguf: the shards hold %d tensors, the first shard's split.tensors.count says %d", len(out.tensors), want))
	}
	out.mmaps = maps
	runtime.SetFinalizer(out, finalizeGGUFMmap)
	return out, nil
}
