package neural

import (
    "testing"
    "github.com/skelterjohn/go.matrix"
)

func TestMeanSquaredError (t *testing.T) {
    var err error
    datum := new(Datum)
    datum.output = matrix.Ones(1,5)
    mse, _ := MeanSquaredError(matrix.Zeros(1,5), datum.output)
    if mse != 1.0 {
        t.Error("MSE != 1.0")
    }
    mse, _ = MeanSquaredError(matrix.Numbers(1, 5, 0.5), datum.output)
    if mse != 0.25 {
        t.Error("MSE != 0.25")
    }
    mse, err = MeanSquaredError(matrix.Zeros(1,4), datum.output)
    if err == nil {
        t.Error("MSE of the wrong size vector didn't return an error")
    }
}
