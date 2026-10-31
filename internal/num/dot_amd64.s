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
