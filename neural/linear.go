package neural

import (
    . "code.google.com/p/biogo.matrix"
    "math/rand"
    "fmt"
)

type Linear struct {
    weights *Dense
    bias *Dense
    Datum
}

func randomWeight () float64 {
    return rand.Float64() * 2 - 1.0
}

func LinearLayer(inputs, outputs int) (layer *Linear) {
    weights, _ := FuncDense(outputs, inputs, 1.0, randomWeight)
    bias, _ := FuncDense(outputs, 1, 1.0, randomWeight)
    return &Linear{weights: weights, bias: bias}
}

func (layer *Linear) Activate(input Matrix) Matrix {
    layer.input = input
    // f(x) = W * x + b  
    output := layer.weights.MulElem(input, nil)
    return output.Add(layer.bias, output)
}

func (layer *Linear) Train(cost Matrix, rate float64) (residual Matrix) {
    // dC/dx = transpose(W) x dC/d(f(x)) 
    weightsT := layer.weights.T(nil)
    residual = weightsT.MulElem(cost, nil)
    weightGradient := cost.MulElem(weightsT, weightsT) // reuse weightsT
    // scale the gradient by the learning rate and update the weights
    weightGradient.Scalar(rate, weightGradient)
    layer.weights.Add(weightGradient, layer.weights)
    // dC/db = dC/d(f(x)) ... which is just the cost
    // scale the gradient by the learning rate and update the bias
    bias_gradient := cost.Scalar(rate, cost) // reuse cost
    layer.bias.Add(bias_gradient, layer.bias)
    return
}

func (layer *Linear) String() string {
    return fmt.Sprintf("<Linear %v + %v>", layer.weights, layer.bias)
}
