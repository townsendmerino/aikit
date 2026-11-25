package pkg

import "testing"

func TestSum(t *testing.T) { _ = Sum([]float32{1}) }

func TestUnrelated(t *testing.T) {}

var sumCases = []func([]float32) float32{Sum}

func TestViaTable(t *testing.T) {
	for _, f := range sumCases {
		_ = f([]float32{1})
	}
}

func viaHelper() { _ = Sum(nil) }

func TestViaHelper(t *testing.T) { viaHelper() }

func TestViaWrapper(t *testing.T) { _ = Wrapper([]float32{1}) }

func TestArm(t *testing.T) { x := []float32{1}; _ = armK(&x[0], 1) }
