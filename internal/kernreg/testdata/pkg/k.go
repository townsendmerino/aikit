package pkg

func sumK(x *float32, n int) float32

func probe() uint32

func Sum(x []float32) float32 { return sumK(&x[0], len(x)) }

type Mat struct{ x []float32 }

func (m *Mat) Rows() int { return len(m.x) }

type inner struct{}

func (inner) Hidden() {}

func Config() int { return 1 }

func Wrapper(x []float32) float32 { return Sum(x) }
