package neural

import (
    "github.com/skelterjohn/go.matrix"
)

type HardTanh struct {
    Datum
}

func HardTanhLayer(inputs int) (layer *HardTanh) {
    return new(HardTanh)
}

func (layer *HardTanh) Activate(input matrix.MatrixRO) (matrix.MatrixRO, error) {
    layer.input = matrix.MakeDenseCopy(input)
    return layer.computeHardTanh(input), nil
}

func (layer *HardTanh) Train(cost matrix.MatrixRO, rate float64) (residual matrix.MatrixRO, err error) {
    return layer.computeHardTanh(cost), nil
}
   
func (layer *HardTanh) computeHardTanh(vector matrix.MatrixRO) matrix.MatrixRO {
    output := matrix.MakeDenseCopy(vector)
    rows, _ := output.GetSize()
    for i := 0; i < rows; i++ {
        v := layer.input.Get(i, 1)
        if v < -1.0 {
            output.Set(i, 1, -1.0)
        } else if v > 1.0 {
            output.Set(i, 1, 1.0)
        }
    }
    return output
}

func String() string {
    return "<HardTanh>"
}
