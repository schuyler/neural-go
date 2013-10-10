package neural

import (
    "github.com/skelterjohn/go.matrix"
)

type LinearLayer struct {
    weights *matrix.DenseMatrix
    bias *matrix.DenseMatrix
    layerBase
}

func MakeLinearLayer(inputs, outputs int, rate float64) (layer *LinearLayer) {
    layer = new(LinearLayer)
    layer.weights = matrix.Normals(inputs, outputs)
    layer.bias = matrix.Normals(1, outputs)
    layer.rate = rate
    return
}

func (layer *LinearLayer) Activate(input matrix.MatrixRO) (output matrix.MatrixRO) {
    var err error
    var product matrix.Matrix
    layer.input = input
    // f(x) = W * x + b  
    product, err = input.Times(layer.weights)
    if err != nil {
        panic(err)
    }
    output, err = product.Plus(layer.bias)
    if err != nil {
        panic(err)
    }
    layer.output = output
    return
}

func (layer *LinearLayer) Train(cost matrix.MatrixRO, rate float64) (residual matrix.MatrixRO) {
    var err error
    var weight_gradient matrix.Matrix
    // dC/dx = transpose(W) x dC/d(f(x)) 
    residual, err = matrix.Transpose(layer.weights).Times(cost)
    if err != nil {
        panic(err)
    }
    // dC/dW = dC/d(f(x)) x transpose(x)
    weight_gradient, err = cost.Times(matrix.Transpose(layer.input))
    if err != nil {
        panic(err)
    }
    // scale the gradient by the learning rate and update the weights
    weight_gradient.Scale(rate)
    err = layer.weights.Add(weight_gradient)
    if err != nil {
        panic(err)
    }
    // dC/db = dC/d(f(x)) ... which is just the cost
    // scale the gradient by the learning rate and update the bias
    bias_gradient := matrix.Scaled(cost, layer.rate)
    err = layer.bias.Add(bias_gradient)
    if err != nil {
        panic(err)
    }
    return
}
