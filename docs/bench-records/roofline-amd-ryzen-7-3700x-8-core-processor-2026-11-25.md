# Roofline census: AMD Ryzen 7 3700X 8-Core Processor, 2026-11-25

Written by `go run -C tools ./roofline` at aikit `5db87ff`, Linux/x86_64, 2026-11-25T16:12:03Z. 3 passes at `-benchtime 300ms`, median ns/op; wall time 1m23s. Single-threaded. Read it against [`roofline-registration.md`](roofline-registration.md).

## Roofs (one core)

| roof | value | how |
|---|--:|---|
| f32 FMA peak | 136.4 GFLOP/s (68.2 GMAC/s f32, 34.1 f64; implies 4.26 GHz) | measured: `measureFMAPeak` |
| int8 dot peak | 69.0 GMAC/s | cited: docs/internal/roofline-2026-08.md § 0, Ryzen 7 3700X |
| streaming read | 21.9 GB/s (32 KiB: 81.4; cross-check passed) | measured: `bytes.IndexByte` over 512 MiB, best of 5 |

## Rows, lowest fraction first

Bytes are each row's operands counted once, not its traffic. A resident row is held to its class's compute roof; a streamed row to the lower of that and its MACs per byte times the streaming roof.

| kernel | class | regime | shape | MiB | ns/op | GB/s | GMAC/s | binding | fraction | |
|---|---|---|---|--:|--:|--:|--:|---|--:|---|
| MatmulAVAcc64Group_G4 | f64 | resident | G4_hd128_depth256 | 0.13 | 156426 | 0.877 | 0.838 | compute | 2% |  |
| MatmulQKAcc64Group_G4 | f64 | resident | G4_hd128_depth256 | 0.13 | 141692 | 0.968 | 0.925 | compute | 3% |  |
| MatmulAVAcc64Group_G4 | f64 | streamed | G4_hd128_depth262144 | 132.00 | 161189334 | 0.859 | 0.833 | stream | 4% | laggard |
| MatmulQKAcc64Group_G4 | f64 | streamed | G4_hd128_depth262144 | 132.00 | 147483404 | 0.939 | 0.91 | stream | 4% | laggard |
| GELUContractInto | stream | streamed | n33554432 | 128.00 | 99150394 | 1.35 | — | stream | 6% | laggard |
| SoftmaxRowContractInto | stream | streamed | n33554432 | 128.00 | 91468769 | 1.47 | — | stream | 7% | laggard |
| F32ToF16Slice | stream | streamed | n33554432 | 128.00 | 86278499 | 1.56 | — | stream | 7% | laggard |
| SiLUContractInto | stream | streamed | n33554432 | 128.00 | 71860280 | 1.87 | — | stream | 9% | laggard |
| AttendTileFused | f32 | resident | np136_hd80 | 0.12 | 461230 | 0.283 | 6.42 | compute | 9% | laggard |
| MatmulBTQ8Into_M1 | f32 | resident | M1_K512_N256 | 0.13 | 16174 | 8.29 | 8.1 | compute | 12% |  |
| DequantizeRowsInt8Into | stream | streamed | n134217728 | 128.00 | 47048192 | 2.85 | — | stream | 13% | laggard |
| MatmulQKAcc64F16 | f64 | resident | G1_hd128_depth512 | 0.13 | 11873 | 11.1 | 5.52 | compute | 16% |  |
| WeightMat.q6k_M1 | int8 | resident | M1_K512_N256 | 0.10 | 11247 | 9.74 | 11.7 | compute | 17% |  |
| WeightMat.q5k_M1 | int8 | resident | M1_K512_N320 | 0.11 | 13935 | 8.23 | 11.8 | compute | 17% |  |
| MatmulBTW4A8F16Into_M1 | int8 | resident | M1_K512_N448_g32 | 0.12 | 19466 | 6.73 | 11.8 | compute | 17% |  |
| AttendTileFusedContractExp | f32 | resident | np136_hd80 | 0.12 | 240468 | 0.543 | 12.3 | compute | 18% | laggard |
| MatmulAVAcc64F16 | f64 | resident | G1_hd128_depth512 | 0.13 | 9800 | 13.4 | 6.69 | compute | 20% |  |
| WeightMat.int4_M1 | int8 | resident | M1_K512_N448_g32_splithalf | 0.12 | 16865 | 7.77 | 13.6 | compute | 20% |  |
| MatmulQKAcc64 | f64 | resident | G1_hd128_depth256 | 0.13 | 4820 | 27.3 | 6.8 | compute | 20% |  |
| WeightMat.q4k_M1 | int8 | resident | M1_K512_N448 | 0.12 | 16356 | 8.01 | 14.0 | compute | 20% |  |
| F16ToF32Slice | stream | streamed | n67108864 | 128.00 | 25121282 | 5.34 | — | stream | 24% | laggard |
| MatmulBT_M1 | f32 | resident | M1_K512_N64 | 0.13 | 1772 | 75.1 | 18.5 | compute | 27% |  |
| MatmulAVAcc64 | f64 | resident | G1_hd128_depth256 | 0.13 | 3151 | 41.8 | 10.4 | compute | 30% |  |
| MatmulQKAcc64 | f64 | streamed | G1_hd128_depth262144 | 128.00 | 19491512 | 6.89 | 1.72 | stream | 31% | laggard |
| MatmulQKAcc64F16 | f64 | streamed | G1_hd128_depth524288 | 128.00 | 19491253 | 6.89 | 3.44 | stream | 31% | laggard |
| MatmulBTW8A8Into_M1 | int8 | resident | M1_K512_N256 | 0.13 | 5491 | 24.4 | 23.9 | compute | 35% |  |
| WeightMat.int4_M32 | int8 | resident | M32_K512_N448_g32_splithalf | 0.19 | 287718 | 0.676 | 25.5 | compute | 37% |  |
| QuantizeRowInt8 | stream | streamed | n33554432 | 128.00 | 16029908 | 8.37 | — | stream | 38% | laggard |
| WeightMat.int4_M1 | int8 | streamed | M1_K4096_N58240_g32_splithalf | 127.98 | 15492545 | 8.66 | 15.4 | stream | 39% | laggard |
| MatmulBTW4A8F16Into_M1 | int8 | streamed | M1_K4096_N58240_g32 | 127.98 | 15463679 | 8.68 | 15.4 | stream | 40% | laggard |
| MatmulBTQ8Into_M1 | f32 | streamed | M1_K4096_N32768 | 128.14 | 14390796 | 9.34 | 9.33 | stream | 43% | laggard |
| MatmulBT_M32 | f32 | streamed | M32_K4096_N8192 | 128.50 | 36850537 | 3.66 | 29.1 | compute | 43% | laggard |
| WeightMat.int4_M32 | int8 | streamed | M32_K4096_N58240_g32_splithalf | 128.47 | 257645778 | 0.523 | 29.6 | compute | 43% | laggard |
| MatmulBTW8A8Into_M32 | int8 | resident | M32_K512_N256 | 0.19 | 114451 | 1.73 | 36.6 | compute | 53% |  |
| MatmulBT_M32 | f32 | resident | M32_K512_N64 | 0.19 | 26236 | 7.49 | 40.0 | compute | 59% |  |
| HammingRows | stream | streamed | n134217728 | 128.00 | 10338398 | 13.0 | — | stream | 59% |  |
| WeightMat.q5k_M1 | int8 | streamed | M1_K4096_N47616 | 127.89 | 10043447 | 13.4 | 19.4 | stream | 61% |  |
| WeightMat.q4k_M1 | int8 | streamed | M1_K4096_N58240 | 127.98 | 9977195 | 13.5 | 23.9 | stream | 61% |  |
| WeightMat.q6k_M1 | int8 | streamed | M1_K4096_N39936 | 127.98 | 9936306 | 13.5 | 16.5 | stream | 62% |  |
| MatmulAVAcc64F16 | f64 | streamed | G1_hd128_depth524288 | 128.00 | 9680585 | 13.9 | 6.93 | stream | 63% |  |
| MatmulBTW8A8Into_M32 | int8 | streamed | M32_K4096_N32768 | 128.62 | 92640814 | 1.46 | 46.4 | compute | 67% |  |
| MatmulBTW8A8Into_M1 | int8 | streamed | M1_K4096_N32768 | 128.14 | 7431798 | 18.1 | 18.1 | stream | 82% |  |
| MatmulAVAcc64 | f64 | streamed | G1_hd128_depth262144 | 128.00 | 6604648 | 20.3 | 5.08 | stream | 93% |  |
| MatmulBT_M1 | f32 | streamed | M1_K4096_N8192 | 128.02 | 5499629 | 24.4 | 6.1 | stream | 111% |  |
| SoftmaxRowContractInto | stream | resident | n32768 | 0.12 | 97069 | 1.35 | — | n/a: no cache roof measured | — |  |
| SiLUContractInto | stream | resident | n32768 | 0.12 | 68126 | 1.92 | — | n/a: no cache roof measured | — |  |
| GELUContractInto | stream | resident | n32768 | 0.12 | 95494 | 1.37 | — | n/a: no cache roof measured | — |  |
| QuantizeRowInt8 | stream | resident | n32768 | 0.12 | 6717 | 19.5 | — | n/a: no cache roof measured | — |  |
| F16ToF32Slice | stream | resident | n65536 | 0.12 | 4361 | 30.1 | — | n/a: no cache roof measured | — |  |
| F32ToF16Slice | stream | resident | n32768 | 0.12 | 75781 | 1.73 | — | n/a: no cache roof measured | — |  |
| DequantizeRowsInt8Into | stream | resident | n131072 | 0.12 | 8314 | 15.8 | — | n/a: no cache roof measured | — |  |
| HammingRows | stream | resident | n131072 | 0.12 | 7924 | 16.5 | — | n/a: no cache roof measured | — |  |
