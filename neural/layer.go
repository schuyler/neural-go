package neural

import (
    "time"
    "math/rand"
    . "code.google.com/p/biogo.matrix"
)

type Layer interface {
    Train(input Matrix) (residual Matrix)
    Activate(input Matrix) (result Matrix)
}

type Datum struct {
    input Matrix
    output Matrix
}

func init() {
    rand.Seed(time.Now().UTC().UnixNano())
}

func MeanSquaredError(expected, observed Matrix) float64 {
    delta := expected.Sub(observed, nil)
    squared_error := 0.0
    mse := func (r, c int, v float64) float64 { 
        squared_error += v * v
        return v
    }
    delta.Apply(mse, nil)
    rows, cols := delta.Dims()
    return squared_error / float64(rows * cols)
}
