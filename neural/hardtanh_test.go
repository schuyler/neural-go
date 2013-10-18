package neural

import (
    . "github.com/smartystreets/goconvey/convey"
    "github.com/skelterjohn/go.matrix"
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
        input := matrix.MakeDenseMatrix([]float64{-2.0, -0.5, 0.5, 2.0}, 4, 1)
        expected := matrix.MakeDenseMatrix([]float64{-1.0, -0.5, 0.5, 1.0}, 4, 1)
        Convey("the output should be the input with a 'HardTanh' applied", func() {
            output, err := layer.Activate(input)
            So(err, ShouldBeNil)
            rows, cols := output.GetSize()
            So(rows, ShouldEqual, 4)
            So(cols, ShouldEqual, 1)
            So(output, ShouldResemble, expected)
        })
    })
}

func TestHardTanhTrain (t *testing.T) {
    Convey("When training a HardTanhLayer", t, func() {
        layer := HardTanhLayer(4)
        input := matrix.MakeDenseMatrix([]float64{-2.0, -0.5, 0.5, 2.0}, 4, 1)
        cost := matrix.MakeDenseMatrix([]float64{0.1, 0.2, 0.3, 0.4}, 4, 1)
        expected := matrix.MakeDenseMatrix([]float64{0.0, 0.2, 0.3, 0.0}, 4, 1)

        Convey("the residual should be the cost with a 'HardTanh' applied based on the input", func() {
            var residual matrix.MatrixRO
            _, err := layer.Activate(input)
            So(err, ShouldBeNil)
            residual, err = layer.Train(cost, 1.0)
            So(err, ShouldBeNil)
            So(residual, ShouldResemble, expected)
        })
    })
}
