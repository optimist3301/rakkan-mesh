#include "textflag.h"

// Exact block dot products for the block-scaled formats (mx.go). Whatever
// the order of the additions, the results are exact, so they equal the
// portable code's bit for bit.

// E2M1 code -> twice its value, as a signed byte: 0 1 2 3 4 6 8 12, negated
// for codes 8-15.
DATA fp4lut<>+0(SB)/8, $0x0c08060403020100
DATA fp4lut<>+8(SB)/8, $0xf4f8fafcfdfeff00
GLOBL fp4lut<>(SB), RODATA|NOPTR, $16

// One FP4 block: 16 packed bytes at off(SI) against 32 int32 at xoff(DI),
// the activation as integers (E4M3 value times 2^9), permuted so that the
// 16 low nibbles (even elements) meet the first 16 and the high nibbles
// (odd elements) the last 16. Leaves eight partial sums in ACC. Uses
// X0-X5, Y2-Y5; X8 holds 0x0f bytes, X9 the table.
#define FP4BLOCK(off, xoff, ACC) \
	VMOVDQU off(SI), X0; \
	VPSRLW $4, X0, X1; \
	VPAND X8, X0, X0; \
	VPAND X8, X1, X1; \
	VPSHUFB X0, X9, X0; \
	VPSHUFB X1, X9, X1; \
	VPMOVSXBD X0, Y2; \
	VPSRLDQ $8, X0, X0; \
	VPMOVSXBD X0, Y3; \
	VPMOVSXBD X1, Y4; \
	VPSRLDQ $8, X1, X1; \
	VPMOVSXBD X1, Y5; \
	VPMULLD xoff+0(DI), Y2, Y2; \
	VPMULLD xoff+32(DI), Y3, Y3; \
	VPMULLD xoff+64(DI), Y4, Y4; \
	VPMULLD xoff+96(DI), Y5, Y5; \
	VPADDD Y3, Y2, Y2; \
	VPADDD Y5, Y4, Y4; \
	VPADDD Y4, Y2, ACC

// Integer sums of four FP4 blocks at a time (nb is a multiple of 4): each
// |sum| < 32·12·229376 < 2^27, exact in int32.
//
// func blockSumsFP4AVX2(w *byte, x *int32, nb int, sums *int32)
TEXT ·blockSumsFP4AVX2(SB), NOSPLIT, $0-32
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ nb+16(FP), CX
	MOVQ sums+24(FP), DX
	MOVL $0x0f0f0f0f, AX
	VMOVD AX, X8
	VPBROADCASTD X8, X8
	VMOVDQU fp4lut<>(SB), X9
blockSumsFP4AVX2loop:
	FP4BLOCK(0, 0, Y10)
	FP4BLOCK(16, 128, Y11)
	FP4BLOCK(32, 256, Y12)
	FP4BLOCK(48, 384, Y13)
	VPHADDD Y11, Y10, Y10
	VPHADDD Y13, Y12, Y12
	VPHADDD Y12, Y10, Y10
	VEXTRACTI128 $1, Y10, X11
	VPADDD X11, X10, X10
	VMOVDQU X10, (DX)
	ADDQ $64, SI
	ADDQ $512, DI
	ADDQ $16, DX
	SUBQ $4, CX
	JNZ blockSumsFP4AVX2loop
	VZEROUPPER
	RET

// Eight E4M3 codes at off(SI) as float64 in Y2 (first four) and Y3: the
// seven magnitude bits placed at bits 20-26 of a float32 read as a value
// 2^-120 times the E4M3 value, normal or subnormal alike, so one exact
// multiplication by 2^120 (Y14) converts them; the sign bit (Y13 masks it)
// moves to bit 31. Y12 masks the magnitude.
#define FP8DECODE(off) \
	VPMOVZXBD off(SI), Y0; \
	VPAND Y13, Y0, Y1; \
	VPSLLD $24, Y1, Y1; \
	VPAND Y12, Y0, Y0; \
	VPSLLD $20, Y0, Y0; \
	VMULPS Y14, Y0, Y0; \
	VPOR Y1, Y0, Y0; \
	VCVTPS2PD X0, Y2; \
	VEXTRACTF128 $1, Y0, X0; \
	VCVTPS2PD X0, Y3

// Exact float64 sums of FP8 blocks: products of E4M3 values are exact, and
// so is any partial sum of 32 of them (multiples of 2^-18 below 2^23).
//
// func blockSumsFP8AVX2(w *byte, x *float64, nb int, sums *float64)
TEXT ·blockSumsFP8AVX2(SB), NOSPLIT, $0-32
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ nb+16(FP), CX
	MOVQ sums+24(FP), DX
	MOVL $0x80, AX
	VMOVD AX, X13
	VPBROADCASTD X13, Y13
	MOVL $0x7f, AX
	VMOVD AX, X12
	VPBROADCASTD X12, Y12
	MOVL $0x7b800000, AX // 2^120
	VMOVD AX, X14
	VPBROADCASTD X14, Y14
blockSumsFP8AVX2loop:
	FP8DECODE(0)
	VMULPD 0(DI), Y2, Y8
	VMULPD 32(DI), Y3, Y9
	FP8DECODE(8)
	VFMADD231PD 64(DI), Y2, Y8
	VFMADD231PD 96(DI), Y3, Y9
	FP8DECODE(16)
	VFMADD231PD 128(DI), Y2, Y8
	VFMADD231PD 160(DI), Y3, Y9
	FP8DECODE(24)
	VFMADD231PD 192(DI), Y2, Y8
	VFMADD231PD 224(DI), Y3, Y9
	VADDPD Y9, Y8, Y8
	VEXTRACTF128 $1, Y8, X9
	VADDPD X9, X8, X8
	VPERMILPD $1, X8, X9
	VADDSD X9, X8, X8
	VMOVSD X8, (DX)
	ADDQ $32, SI
	ADDQ $256, DI
	ADDQ $8, DX
	DECQ CX
	JNZ blockSumsFP8AVX2loop
	VZEROUPPER
	RET
