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
    output := matrix.MakeDenseCopy(input)
    rows, _ := output.GetSize()
    for i := 0; i < rows; i++ {
        v := layer.input.Get(i, 0)
        if v < -1.0 {
            output.Set(i, 0, -1.0)
        } else if v > 1.0 {
            output.Set(i, 0, 1.0)
        }
    }
    return output, nil
}

func (layer *HardTanh) Train(cost matrix.MatrixRO, rate float64) (residual matrix.MatrixRO, err error) {
    output := matrix.MakeDenseCopy(cost)
    rows, _ := output.GetSize()
    for i := 0; i < rows; i++ {
        v := layer.input.Get(i, 0)
        if v < -1.0 {
            output.Set(i, 0, 0.0)
        } else if v > 1.0 {
            output.Set(i, 0, 0.0)
        }
    }
    return output, nil
}

func String() string {
    return "<HardTanh>"
}
