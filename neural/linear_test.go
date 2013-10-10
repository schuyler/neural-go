package neural

import (
    "testing"
)

func TestMakeLinearLayer (t *testing.T) {
    MakeLinearLayer(3, 4)
}

func TestActivate (t *testing.T) {
    layer := MakeLinearLayer(3, 1, 0.1)
    input := matrix.MakeDenseMatrix(&[]float64{0.0, 0.0, 0.0})

}
