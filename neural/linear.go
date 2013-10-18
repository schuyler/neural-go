package neural

import (
    "github.com/skelterjohn/go.matrix"
    "fmt"
)

type Linear struct {
    weights *matrix.DenseMatrix
    bias *matrix.DenseMatrix
    layerBase
}

func LinearLayer(inputs, outputs int) (layer *Linear) {
    return &Linear{
        weights: matrix.Normals(outputs, inputs), 
        bias: matrix.Normals(outputs, 1)}
}

func (layer *Linear) Activate(input matrix.MatrixRO) (matrix.MatrixRO, error) {
    var output matrix.Matrix
    var err error
    layer.input = matrix.MakeDenseCopy(input)
    // f(x) = W * x + b  
    output, err = layer.weights.Times(input)
    if err != nil {
        return nil, err
    }
    err = output.Add(layer.bias)
    if err != nil {
        return nil, err
    }
    return output, nil
}

func (layer *Linear) Train(cost matrix.MatrixRO, rate float64) (residual matrix.MatrixRO, err error) {
    var weight_gradient matrix.Matrix
    // dC/dx = transpose(W) x dC/d(f(x)) 
    residual, err = layer.weights.Transpose().Times(cost)
    if err != nil {
        return nil, err
    }
    // dC/dW = dC/d(f(x)) x transpose(x)
    weight_gradient, err = cost.Times(matrix.Transpose(layer.input))
    if err != nil {
        return nil, err
    }
    // scale the gradient by the learning rate and update the weights
    weight_gradient.Scale(rate)
    err = layer.weights.Add(weight_gradient)
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

func (layer *Linear) String() string {
    return fmt.Sprintf("<Linear %v + %v>", layer.weights, layer.bias)
}
