package pkg

func SumAMD64(x []float32) float32 { return sumK(&x[0], len(x)) }

func armK(x *float32, n int) float32
