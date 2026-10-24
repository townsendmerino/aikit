package embed

// Q4KRaw returns tensor name's raw GGUF Q4_K super-blocks (144 bytes per 256 elements, ggml's
// block_q4_K layout, rows contiguous along dims[0]) and its dims, when the tensor is stored as Q4_K.
// ok is false, with a nil error, for a tensor of any other type. The bytes alias the file mapping,
// so the caller keeps the GGUFFile open while it uses them. For a consumer that multiplies Q4_K
// natively (linalg.WrapQ4K) instead of dequantizing: see goinfer's
// docs/tasks/task-int4-weight-quality-2026-09.md.
func (g *GGUFFile) Q4KRaw(name string) (dims []int, raw []byte, ok bool, err error) {
	return g.kQuantRaw(name, ggmlTypeQ4_K)
}

// Q5KRaw is Q4KRaw for a tensor stored as Q5_K: ggml's block_q5_K, 176 bytes per 256 elements (d, dmin, scales[12],
// qh[32], qs[128]).
func (g *GGUFFile) Q5KRaw(name string) (dims []int, raw []byte, ok bool, err error) {
	return g.kQuantRaw(name, ggmlTypeQ5_K)
}

// Q6KRaw is Q4KRaw for a tensor stored as Q6_K: ggml's block_q6_K, 210 bytes per 256 elements (ql[128], qh[64],
// scales[16] as int8, d). A block is not 4-byte aligned.
func (g *GGUFFile) Q6KRaw(name string) (dims []int, raw []byte, ok bool, err error) {
	return g.kQuantRaw(name, ggmlTypeQ6_K)
}

// kQuantRaw returns tensor name's raw blocks when it is stored as typ, aliasing the file mapping.
func (g *GGUFFile) kQuantRaw(name string, typ uint32) (dims []int, raw []byte, ok bool, err error) {
	info, found := g.tensors[name]
	if !found || info.typ != typ {
		return nil, nil, false, nil
	}
	// RowDequantizer validates dims against the data section (overflow-safe); reuse it.
	dims, _, err = g.RowDequantizer(name)
	if err != nil {
		return nil, nil, false, err
	}
	n := 1
	for _, d := range dims {
		n *= d
	}
	raw, err = g.tensorBytes(info, n)
	if err != nil {
		return nil, nil, false, err
	}
	return dims, raw, true, nil
}
