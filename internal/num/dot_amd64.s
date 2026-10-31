#include "textflag.h"

// Four 8-lane accumulators: Y0 holds sums 0-7, Y1 8-15, Y2 16-23, Y3
// 24-31. Products are rounded by VMULPS before VADDPS adds them: no fused
// multiply-add, matching the portable code bit for bit. n is a multiple of
// 32; the sums are loaded from and stored back to s.

// func dot32AVX2(a, b *float32, n int, s *[32]float32)
TEXT ·dot32AVX2(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX
	MOVQ s+24(FP), DX
	VMOVUPS (DX), Y0
	VMOVUPS 32(DX), Y1
	VMOVUPS 64(DX), Y2
	VMOVUPS 96(DX), Y3
	TESTQ CX, CX
	JZ done32
loop32:
	VMOVUPS (SI), Y4
	VMULPS (DI), Y4, Y4
	VADDPS Y4, Y0, Y0
	VMOVUPS 32(SI), Y5
	VMULPS 32(DI), Y5, Y5
	VADDPS Y5, Y1, Y1
	VMOVUPS 64(SI), Y6
	VMULPS 64(DI), Y6, Y6
	VADDPS Y6, Y2, Y2
	VMOVUPS 96(SI), Y7
	VMULPS 96(DI), Y7, Y7
	VADDPS Y7, Y3, Y3
	ADDQ $128, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JNZ loop32
done32:
	VMOVUPS Y0, (DX)
	VMOVUPS Y1, 32(DX)
	VMOVUPS Y2, 64(DX)
	VMOVUPS Y3, 96(DX)
	VZEROUPPER
	RET

// bfloat16 weights widen exactly: zero-extend to 32 bits, shift left 16.

// func dotBF16AVX2(w *uint16, x *float32, n int, s *[32]float32)
TEXT ·dotBF16AVX2(SB), NOSPLIT, $0-32
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ n+16(FP), CX
	MOVQ s+24(FP), DX
	VMOVUPS (DX), Y0
	VMOVUPS 32(DX), Y1
	VMOVUPS 64(DX), Y2
	VMOVUPS 96(DX), Y3
	TESTQ CX, CX
	JZ donebf
loopbf:
	VPMOVZXWD (SI), Y4
	VPSLLD $16, Y4, Y4
	VMULPS (DI), Y4, Y4
	VADDPS Y4, Y0, Y0
	VPMOVZXWD 16(SI), Y5
	VPSLLD $16, Y5, Y5
	VMULPS 32(DI), Y5, Y5
	VADDPS Y5, Y1, Y1
	VPMOVZXWD 32(SI), Y6
	VPSLLD $16, Y6, Y6
	VMULPS 64(DI), Y6, Y6
	VADDPS Y6, Y2, Y2
	VPMOVZXWD 48(SI), Y7
	VPSLLD $16, Y7, Y7
	VMULPS 96(DI), Y7, Y7
	VADDPS Y7, Y3, Y3
	ADDQ $64, SI
	ADDQ $128, DI
	SUBQ $32, CX
	JNZ loopbf
donebf:
	VMOVUPS Y0, (DX)
	VMOVUPS Y1, 32(DX)
	VMOVUPS Y2, 64(DX)
	VMOVUPS Y3, 96(DX)
	VZEROUPPER
	RET

// Integer block sums, four blocks at a time (nb is a multiple of 4): for
// each block of 32, sum w[j]*x[j] exactly. The signed-by-signed product
// goes through VPMADDUBSW, which multiplies unsigned by signed bytes: |w|
// times x carrying w's sign. Pairs of products fit in 16 bits
// (2·128·127 < 32768) and VPMADDWD widens them to 32; two rounds of
// VPHADDD and a final add reduce each block's eight partial sums. Q4_0
// blocks unpack to 32 signed bytes in order: the low nibbles of the 16
// bytes are weights 0-15, the high nibbles weights 16-31.

// func blockSumsQ8AVX2(w *int8, x *int8, nb int, sums *int32)
TEXT ·blockSumsQ8AVX2(SB), NOSPLIT, $0-32
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ nb+16(FP), CX
	MOVQ sums+24(FP), DX
	MOVL $0x00010001, AX
	VMOVD AX, X7
	VPBROADCASTD X7, Y7
blockSumsQ8AVX2loop:
	VMOVDQU 0(SI), Y0
	VMOVDQU 0(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y10
	VMOVDQU 32(SI), Y0
	VMOVDQU 32(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y11
	VMOVDQU 64(SI), Y0
	VMOVDQU 64(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y12
	VMOVDQU 96(SI), Y0
	VMOVDQU 96(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y13
	VPHADDD Y11, Y10, Y10
	VPHADDD Y13, Y12, Y12
	VPHADDD Y12, Y10, Y10
	VEXTRACTI128 $1, Y10, X11
	VPADDD X11, X10, X10
	VMOVDQU X10, (DX)
	ADDQ $128, SI
	ADDQ $128, DI
	ADDQ $16, DX
	SUBQ $4, CX
	JNZ blockSumsQ8AVX2loop
	VZEROUPPER
	RET

// func blockSumsQ4AVX2(w *byte, x *int8, nb int, sums *int32)
TEXT ·blockSumsQ4AVX2(SB), NOSPLIT, $0-32
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ nb+16(FP), CX
	MOVQ sums+24(FP), DX
	MOVL $0x00010001, AX
	VMOVD AX, X7
	VPBROADCASTD X7, Y7
	MOVL $0x0f0f0f0f, AX
	VMOVD AX, X8
	VPBROADCASTD X8, Y8
	MOVL $0x08080808, AX
	VMOVD AX, X9
	VPBROADCASTD X9, Y9
blockSumsQ4AVX2loop:
	VMOVDQU 0(SI), X0
	VPSRLW $4, X0, X1
	VINSERTI128 $1, X1, Y0, Y0
	VPAND Y8, Y0, Y0
	VPSUBB Y9, Y0, Y0
	VMOVDQU 0(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y10
	VMOVDQU 16(SI), X0
	VPSRLW $4, X0, X1
	VINSERTI128 $1, X1, Y0, Y0
	VPAND Y8, Y0, Y0
	VPSUBB Y9, Y0, Y0
	VMOVDQU 32(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y11
	VMOVDQU 32(SI), X0
	VPSRLW $4, X0, X1
	VINSERTI128 $1, X1, Y0, Y0
	VPAND Y8, Y0, Y0
	VPSUBB Y9, Y0, Y0
	VMOVDQU 64(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y12
	VMOVDQU 48(SI), X0
	VPSRLW $4, X0, X1
	VINSERTI128 $1, X1, Y0, Y0
	VPAND Y8, Y0, Y0
	VPSUBB Y9, Y0, Y0
	VMOVDQU 96(DI), Y1
	VPSIGNB Y0, Y0, Y2
	VPSIGNB Y0, Y1, Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y7, Y4, Y13
	VPHADDD Y11, Y10, Y10
	VPHADDD Y13, Y12, Y12
	VPHADDD Y12, Y10, Y10
	VEXTRACTI128 $1, Y10, X11
	VPADDD X11, X10, X10
	VMOVDQU X10, (DX)
	ADDQ $64, SI
	ADDQ $128, DI
	ADDQ $16, DX
	SUBQ $4, CX
	JNZ blockSumsQ4AVX2loop
	VZEROUPPER
	RET
