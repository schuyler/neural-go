package neural

import (
    "time"
    "math/rand"
    "github.com/skelterjohn/go.matrix"
)

type Layer interface {
    Train(input matrix.Matrix, err matrix.Matrix) (residual matrix.Matrix)
    Activate(input matrix.Matrix) (result matrix.Matrix)
}

type Datum struct {
    input matrix.MatrixRO
    output matrix.MatrixRO
}

func initialize() {
    rand.Seed(time.Now().UTC().UnixNano())
}

func (layer *layerBase) MeanSquaredError(expected matrix.MatrixRO) (float64, error) {
    delta, err := layer.output.Minus(expected)
    if err != nil {
        return 0.0, err
    }
    squared_error := 0.0
    for i := 0; i < delta.Rows(); i++ {
        for j := 0; j < delta.Cols(); j++ {
            value := delta.Get(i, j)
            squared_error += value * value
        }
    }
    return squared_error / float64(delta.NumElements()), nil
}
