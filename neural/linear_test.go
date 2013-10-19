package neural

import (
    . "github.com/smartystreets/goconvey/convey"
    "code.google.com/p/biogo.matrix"
    "testing"
)

func TestLinearLayer (t *testing.T) {
    Convey("When creating a LinearLayer", t, func() {
        Convey("the constructor should succeed", func() {
            So(func() { LinearLayer(3, 1) }, ShouldNotPanic)
        })
    })
}

func TestActivate (t *testing.T) {
    Convey("When activating a LinearLayer with a unit input", t, func() {
        input, _ := matrix.NewDense([][]float64{{1, 1, 1}})
        expected, _:=  matrix.NewDense([][]float64{{1}})
        bias, _ := matrix.ZeroDense(1,1)
        layer := &Linear{weights: input.Dense(nil), bias: bias}
        Convey("the output should be the product of the input and weights plus the bias", func() {
            output := layer.Activate(input)
            rows, cols := output.Dims()
            So(rows, ShouldEqual, 1)
            So(cols, ShouldEqual, 1)
            So(output, ShouldResemble, expected)
        })
    })
}

func TestTrain (t *testing.T) {
    Convey("When training a LinearLayer", t, func() {
        input, _ := matrix.NewDense([][]float64{{1, 1, 1}})
        expected, _ := matrix.NewDense([][]float64{{3, 3}})
        layer := LinearLayer(3, 2)

        Convey("the MSE should improve at every epoch", func() {
            epochs := 20
            for delta := 1.0; epochs > 0 && delta > 0.01; epochs-- {
                output := layer.Activate(input)
                cost := expected.Sub(output, nil)
                layer.Train(cost, 1.0)
                So(delta, ShouldBeGreaterThan, cost.At(0,0))
                delta = cost.At(0, 0)
            }
            So(epochs, ShouldBeGreaterThan, 0)
        })
    })
}
