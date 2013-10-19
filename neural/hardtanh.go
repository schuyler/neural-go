package neural

import (
    . "code.google.com/p/biogo.matrix"
)

type HardTanh struct {
    Datum
}

func HardTanhLayer(inputs int) (layer *HardTanh) {
    return new(HardTanh)
}

func (layer *HardTanh) Activate(input Matrix) Matrix {
    layer.input = input
    output := input.Dense(nil)
    rows, _ := output.Dims()
    for i := 0; i < rows; i++ {
        v := layer.input.At(i, 0)
        if v < -1.0 {
            output.Set(i, 0, -1.0)
        } else if v > 1.0 {
            output.Set(i, 0, 1.0)
        }
    }
    return output
}

func (layer *HardTanh) Train(cost Matrix, rate float64) (residual Matrix) {
    output := cost.Dense(nil)
    rows, _ := output.Dims()
    for i := 0; i < rows; i++ {
        v := layer.input.At(i, 0)
        if v < -1.0 || v > 1.0 {
            output.Set(i, 0, 0.0)
        }
    }
    return output
}

func String() string {
    return "<HardTanh>"
}
