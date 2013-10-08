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

type layerBase struct {
    input matrix.MatrixRO
    output matrix.MatrixRO
    rate float64
}

func initialize() {
    rand.Seed(time.Now().UTC().UnixNano())
}
