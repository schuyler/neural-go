package neural

import (
    "testing"
    "github.com/skelterjohn/go.matrix"
)

func TestMeanSquaredError (t *testing.T) {
    layer := new(layerBase)
    layer.output = matrix.Ones(1,5)
    mse := layer.MeanSquaredError(matrix.Zeros(1,5))
    if mse != 1.0 {
        t.Fail()
    }
}
