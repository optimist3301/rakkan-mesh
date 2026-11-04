package num

import "math"

// Erf is the error function: fdlibm's s_erf.c, as Go's math.Erf has it,
// with every product rounded before it is added. Go fuses a·b+c on arm64
// and some other processors, so math.Erf's results depend on where it
// runs; this does not.
func Erf(x float64) float64 {
	const (
		veryTiny = 2.848094538889218e-306 // 0x0080000000000000
		small    = 1.0 / (1 << 28)        // 2**-28
	)
	switch {
	case x != x:
		return x
	case math.IsInf(x, 1):
		return 1
	case math.IsInf(x, -1):
		return -1
	}
	sign := false
	if x < 0 {
		x = -x
		sign = true
	}
	if x < 0.84375 {
		var temp float64
		if x < small {
			if x < veryTiny {
				temp = 0.125 * (float64(8.0*x) + float64(efx8*x))
			} else {
				temp = x + float64(efx*x)
			}
		} else {
			z := float64(x * x)
			r := horner(z, pp0, pp1, pp2, pp3, pp4)
			s := horner(z, 1, qq1, qq2, qq3, qq4, qq5)
			y := r / s
			temp = x + float64(x*y)
		}
		if sign {
			return -temp
		}
		return temp
	}
	if x < 1.25 {
		s := x - 1
		p := horner(s, pa0, pa1, pa2, pa3, pa4, pa5, pa6)
		q := horner(s, 1, qa1, qa2, qa3, qa4, qa5, qa6)
		if sign {
			return -erx - p/q
		}
		return erx + p/q
	}
	if x >= 6 {
		if sign {
			return -1
		}
		return 1
	}
	s := 1 / float64(x*x)
	var r, ss float64
	if x < 1/0.35 {
		r = horner(s, ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7)
		ss = horner(s, 1, sa1, sa2, sa3, sa4, sa5, sa6, sa7, sa8)
	} else {
		r = horner(s, rb0, rb1, rb2, rb3, rb4, rb5, rb6)
		ss = horner(s, 1, sb1, sb2, sb3, sb4, sb5, sb6, sb7)
	}
	z := math.Float64frombits(math.Float64bits(x) & 0xffffffff00000000) // x to 20 bits
	e := float64(Exp(float64(-z*z)-0.5625) * Exp(float64((z-x)*(z+x))+r/ss))
	if sign {
		return e/x - 1
	}
	return 1 - e/x
}

// horner is c[0] + x·(c[1] + x·(… + x·c[n])), each product rounded.
func horner(x float64, c ...float64) float64 {
	v := c[len(c)-1]
	for i := len(c) - 2; i >= 0; i-- {
		v = c[i] + float64(x*v)
	}
	return v
}

// fdlibm s_erf.c's coefficients.
const (
	erx = 8.45062911510467529297e-01 // 0x3FEB0AC160000000
	// Coefficients for approximation to  erf in [0, 0.84375]
	efx  = 1.28379167095512586316e-01  // 0x3FC06EBA8214DB69
	efx8 = 1.02703333676410069053e+00  // 0x3FF06EBA8214DB69
	pp0  = 1.28379167095512558561e-01  // 0x3FC06EBA8214DB68
	pp1  = -3.25042107247001499370e-01 // 0xBFD4CD7D691CB913
	pp2  = -2.84817495755985104766e-02 // 0xBF9D2A51DBD7194F
	pp3  = -5.77027029648944159157e-03 // 0xBF77A291236668E4
	pp4  = -2.37630166566501626084e-05 // 0xBEF8EAD6120016AC
	qq1  = 3.97917223959155352819e-01  // 0x3FD97779CDDADC09
	qq2  = 6.50222499887672944485e-02  // 0x3FB0A54C5536CEBA
	qq3  = 5.08130628187576562776e-03  // 0x3F74D022C4D36B0F
	qq4  = 1.32494738004321644526e-04  // 0x3F215DC9221C1A10
	qq5  = -3.96022827877536812320e-06 // 0xBED09C4342A26120
	// Coefficients for approximation to  erf  in [0.84375, 1.25]
	pa0 = -2.36211856075265944077e-03 // 0xBF6359B8BEF77538
	pa1 = 4.14856118683748331666e-01  // 0x3FDA8D00AD92B34D
	pa2 = -3.72207876035701323847e-01 // 0xBFD7D240FBB8C3F1
	pa3 = 3.18346619901161753674e-01  // 0x3FD45FCA805120E4
	pa4 = -1.10894694282396677476e-01 // 0xBFBC63983D3E28EC
	pa5 = 3.54783043256182359371e-02  // 0x3FA22A36599795EB
	pa6 = -2.16637559486879084300e-03 // 0xBF61BF380A96073F
	qa1 = 1.06420880400844228286e-01  // 0x3FBB3E6618EEE323
	qa2 = 5.40397917702171048937e-01  // 0x3FE14AF092EB6F33
	qa3 = 7.18286544141962662868e-02  // 0x3FB2635CD99FE9A7
	qa4 = 1.26171219808761642112e-01  // 0x3FC02660E763351F
	qa5 = 1.36370839120290507362e-02  // 0x3F8BEDC26B51DD1C
	qa6 = 1.19844998467991074170e-02  // 0x3F888B545735151D
	// Coefficients for approximation to  erfc in [1.25, 1/0.35]
	ra0 = -9.86494403484714822705e-03 // 0xBF843412600D6435
	ra1 = -6.93858572707181764372e-01 // 0xBFE63416E4BA7360
	ra2 = -1.05586262253232909814e+01 // 0xC0251E0441B0E726
	ra3 = -6.23753324503260060396e+01 // 0xC04F300AE4CBA38D
	ra4 = -1.62396669462573470355e+02 // 0xC0644CB184282266
	ra5 = -1.84605092906711035994e+02 // 0xC067135CEBCCABB2
	ra6 = -8.12874355063065934246e+01 // 0xC054526557E4D2F2
	ra7 = -9.81432934416914548592e+00 // 0xC023A0EFC69AC25C
	sa1 = 1.96512716674392571292e+01  // 0x4033A6B9BD707687
	sa2 = 1.37657754143519042600e+02  // 0x4061350C526AE721
	sa3 = 4.34565877475229228821e+02  // 0x407B290DD58A1A71
	sa4 = 6.45387271733267880336e+02  // 0x40842B1921EC2868
	sa5 = 4.29008140027567833386e+02  // 0x407AD02157700314
	sa6 = 1.08635005541779435134e+02  // 0x405B28A3EE48AE2C
	sa7 = 6.57024977031928170135e+00  // 0x401A47EF8E484A93
	sa8 = -6.04244152148580987438e-02 // 0xBFAEEFF2EE749A62
	// Coefficients for approximation to  erfc in [1/.35, 28]
	rb0 = -9.86494292470009928597e-03 // 0xBF84341239E86F4A
	rb1 = -7.99283237680523006574e-01 // 0xBFE993BA70C285DE
	rb2 = -1.77579549177547519889e+01 // 0xC031C209555F995A
	rb3 = -1.60636384855821916062e+02 // 0xC064145D43C5ED98
	rb4 = -6.37566443368389627722e+02 // 0xC083EC881375F228
	rb5 = -1.02509513161107724954e+03 // 0xC09004616A2E5992
	rb6 = -4.83519191608651397019e+02 // 0xC07E384E9BDC383F
	sb1 = 3.03380607434824582924e+01  // 0x403E568B261D5190
	sb2 = 3.25792512996573918826e+02  // 0x40745CAE221B9F0A
	sb3 = 1.53672958608443695994e+03  // 0x409802EB189D5118
	sb4 = 3.19985821950859553908e+03  // 0x40A8FFB7688C246A
	sb5 = 2.55305040643316442583e+03  // 0x40A3F219CEDF3BE6
	sb6 = 4.74528541206955367215e+02  // 0x407DA874E79FE763
	sb7 = -2.24409524465858183362e+01 // 0xC03670E242712D62
)
