// Package bitvec holds aikit's binary-vector kernels: sign-bit packing of float32 rows into uint64 words (PackSignBits,
// PackSignBitsRow, PackedWords) and the Hamming distance of one packed query against many packed rows (HammingRows),
// the prefilter behind ann.FlatBinary. HammingRows uses POPCNT on amd64 where the CPU has it (internal/cpufeat), NEON on
// every arm64, and a portable loop otherwise. Every writer's doc comment states its output contract.
package bitvec
