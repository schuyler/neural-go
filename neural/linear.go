package neural

import (
    "github.com/skelterjohn/go.matrix"
)

type LinearLayer struct {
    weights *matrix.DenseMatrix
    bias *matrix.DenseMatrix
    layerBase
}

func MakeLinearLayer(inputs, outputs int) (layer *LinearLayer) {
    return &LinearLayer{
        weights: matrix.Normals(inputs, outputs), 
        bias: matrix.Normals(1, outputs)}
}

func (layer *LinearLayer) Activate(input matrix.MatrixRO) (matrix.MatrixRO, error) {
    var product, output matrix.Matrix
    var err error
    layer.input = input
    // f(x) = W * x + b  
    product, err = input.Times(layer.weights)
    if err != nil {
        return nil, err
    }
    output, err = product.Plus(layer.bias)
    if err != nil {
        return nil, err
    }
    layer.output = output
    return output, nil
}

func (layer *LinearLayer) Train(cost matrix.MatrixRO, rate float64) (residual matrix.MatrixRO, err error) {
    var weight_gradient matrix.Matrix
    // dC/dx = transpose(W) x dC/d(f(x)) 
    residual, err = layer.weights.Times(cost)
    if err != nil {
        return nil, err
    }
    // dC/dW = dC/d(f(x)) x transpose(x)
    weight_gradient, err = cost.Times(layer.input)
    if err != nil {
        return nil, err
    }
    // scale the gradient by the learning rate and update the weights
    weight_gradient.Scale(rate)
    err = layer.weights.Add(matrix.Transpose(weight_gradient))
    if err != nil {
        return nil, err
    }
    // dC/db = dC/d(f(x)) ... which is just the cost
    // scale the gradient by the learning rate and update the bias
    bias_gradient := matrix.Scaled(cost, rate)
    err = layer.bias.Add(bias_gradient)
    if err != nil {
        return nil, err
    }
    return residual, nil
}
