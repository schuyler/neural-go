package neural

import (
    "testing"
    "code.google.com/p/biogo.matrix"
)

func TestMeanSquaredError (t *testing.T) {
    var (
        zero *matrix.Dense
        point5 *matrix.Dense
    )
    datum := new(Datum)
    datum.output, _ = matrix.NewDense([][]float64{{1, 1, 1, 1, 1}})
    zero, _ = matrix.NewDense([][]float64{{0, 0, 0, 0, 0}})
    point5, _ = matrix.NewDense([][]float64{{0.5, 0.5, 0.5, 0.5, 0.5}})
    mse := MeanSquaredError(zero, datum.output)
    if mse != 1.0 {
        t.Error("MSE != 1.0")
    }
    mse = MeanSquaredError(point5, datum.output)
    if mse != 0.25 {
        t.Error("MSE != 0.25")
    }
}
