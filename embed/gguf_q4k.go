package embed

// Q4KRaw returns tensor name's raw GGUF Q4_K super-blocks (144 bytes per 256 elements, ggml's
// block_q4_K layout, rows contiguous along dims[0]) and its dims, when the tensor is stored as Q4_K.
// ok is false, with a nil error, for a tensor of any other type. The bytes alias the file mapping,
// so the caller keeps the GGUFFile open while it uses them. For a consumer that multiplies Q4_K
// natively (linalg.WrapQ4K) instead of dequantizing: see goinfer's
// docs/tasks/task-int4-weight-quality-2026-09.md.
func (g *GGUFFile) Q4KRaw(name string) (dims []int, raw []byte, ok bool, err error) {
	info, found := g.tensors[name]
	if !found || info.typ != ggmlTypeQ4_K {
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
