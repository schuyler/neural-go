package neural

import (
    "testing"
    "github.com/skelterjohn/go.matrix"
)

func TestMeanSquaredError (t *testing.T) {
    var err error
    layer := new(layerBase)
    layer.output = matrix.Ones(1,5)
    mse, _ := layer.MeanSquaredError(matrix.Zeros(1,5))
    if mse != 1.0 {
        t.Error("MSE != 1.0")
    }
    mse, _ = layer.MeanSquaredError(matrix.Numbers(1, 5, 0.5))
    if mse != 0.25 {
        t.Error("MSE != 0.25")
    }
    mse, err = layer.MeanSquaredError(matrix.Zeros(1,4))
    if err == nil {
        t.Error("MSE of the wrong size vector didn't return an error")
    }
}
