package pkg

func sumK(x *float32, n int) float32

func probe() uint32

func Sum(x []float32) float32 { return sumK(&x[0], len(x)) }
