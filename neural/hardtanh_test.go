package neural

import (
    . "github.com/smartystreets/goconvey/convey"
    "code.google.com/p/biogo.matrix"
    "testing"
)

func TestHardTanh (t *testing.T) {
    Convey("When creating a HardTanhLayer", t, func() {
        Convey("the constructor should succeed", func() {
            So(func() { HardTanhLayer(3) }, ShouldNotPanic)
        })
    })
}

func TestHardTanhActivate (t *testing.T) {
    Convey("When activating a HardTanhLayer", t, func() {
        layer := HardTanhLayer(4)
        input, _ := matrix.NewDense([][]float64{{-2.0, -0.5, 0.5, 2.0}})
        expected, _ := matrix.NewDense([][]float64{{-1.0, -0.5, 0.5, 1.0}})
        Convey("the output should be the input with a 'HardTanh' applied", func() {
            output := layer.Activate(input)
            rows, cols := output.Dims()
            So(rows, ShouldEqual, 4)
            So(cols, ShouldEqual, 1)
            So(output, ShouldResemble, expected)
        })
    })
}

func TestHardTanhTrain (t *testing.T) {
    Convey("When training a HardTanhLayer", t, func() {
        layer := HardTanhLayer(4)
        input, _ := matrix.NewDense([][]float64{{-2.0, -0.5, 0.5, 2.0}})
        cost, _ := matrix.NewDense([][]float64{{0.1, 0.2, 0.3, 0.4}})
        expected, _ := matrix.NewDense([][]float64{{0.0, 0.2, 0.3, 0.0}})

        Convey("the residual should be the cost with a 'HardTanh' applied based on the input", func() {
            layer.Activate(input)
            residual := layer.Train(cost, 1.0)
            So(residual, ShouldResemble, expected)
        })
    })
}
