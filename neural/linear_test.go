package neural

import (
    . "github.com/smartystreets/goconvey/convey"
    "github.com/skelterjohn/go.matrix"
    "testing"
)

func TestMakeLinearLayer (t *testing.T) {
    Convey("When creating a new LinearLayer", t, func() {
        Convey("the constructor should succeed", func() {
            So(func() { MakeLinearLayer(3, 1) }, ShouldNotPanic)
        })
    })
}

func TestActivate (t *testing.T) {
    Convey("When activating a LinearLayer with a unit input", t, func() {
        layer := &LinearLayer{weights: matrix.Ones(3, 1), bias: matrix.Zeros(1, 1)}
        input := matrix.Ones(1, 3)
        Convey("the output should be the product of the input and weights plus the bias", func() {
            output, err := layer.Activate(input)
            So(err, ShouldBeNil)
            rows, cols := output.GetSize()
            So(rows, ShouldEqual, 1)
            So(cols, ShouldEqual, 1)
            So(output, ShouldResemble, matrix.Numbers(1, 1, 3))
        })
    })
}

func TestTrain (t *testing.T) {
    Convey("When training a LinearLayer", t, func() {
        layer := MakeLinearLayer(3, 1)
        Convey("the MSE should improve at every epoch", func() {
            input := matrix.Ones(1, 3) // {1, 1, 1}
            expected := matrix.Numbers(1, 1, 3.0) // {3}
            epochs := 20
            for delta := 1.0; epochs > 0 && delta > 0.01; epochs-- {
                output, _ := layer.Activate(input)
                cost, _ := expected.Minus(output)
                _, err := layer.Train(cost, 0.1)
                So(err, ShouldBeNil)
                So(delta, ShouldBeGreaterThan, cost.Get(0,0))
                delta = cost.Get(0, 0)
            }
            So(epochs, ShouldBeGreaterThan, 0)
        })
    })
}
