package neural
import (
    "math"
    "rand"
    "time"
    "fmt"
)

type Layer struct {
    weight [][]float64
    bias []float64
    value []float64
}

type Network struct {
    hidden *Layer 
    output *Layer
}

func randomWeight () float64 {
    return rand.Float64() * 2.0 - 1.0
}

func newLayer(inputs int, nodes int) (layer *Layer) {
    layer = new(Layer)
    layer.weight = make([][]float64, nodes)
    for i := 0; i < nodes; i++ {
        layer.weight[i] = make([]float64, inputs)
        for j := 0; j < inputs; j++ {
            layer.weight[i][j] = randomWeight()
        }
    }
    layer.bias = make([]float64, nodes)
    for i := 0; i < nodes; i++ {
        layer.bias[i] = randomWeight()
    }
    layer.value = make([]float64, nodes)
    return
}

func NewNetwork(inputs int, hiddens int, outputs int) (net *Network) {
    net = new(Network)
    net.hidden = newLayer(inputs, hiddens)
    net.output = newLayer(hiddens, outputs)
    return
}

func (layer *Layer) feedforward(input []float64) []float64 {
    for i := 0; i < len(layer.value); i++ {
        sum := layer.bias[i]
        for j := 0; j < len(input); j++ {
            sum += layer.weight[i][j] * input[j]
        }
        layer.value[i] = 1.0 / (1.0 + math.Pow(math.E, -sum))
    }
    return layer.value
}

func (net *Network) Activate(input []float64) (result []float64) {
    hidden := net.hidden.feedforward(input)
    output := net.output.feedforward(hidden)
    result = make([]float64, len(output))
    copy(result, output)
    return
}

func (layer *Layer) backpropagate (input []float64, error []float64, rate float64) (residual []float64) {
    residual = make([]float64, len(layer.weight[0]))
    for i, weight := range layer.weight {
        delta := error[i] * layer.value[i] * (1.0 - layer.value[i])
        for j := 0; j < len(weight); j++ {
            residual[j] += delta * weight[j]
            weight[j] += rate * delta * input[j]
        }
        layer.bias[i] += rate * delta 
    }
    return
}

func (net *Network) Train(input []float64, expected []float64, rate float64) {
    error := make([]float64, len(net.output.value))
    for i := 0; i < len(error); i++ {
        error[i] = expected[i] - net.output.value[i]
    }
    residual := net.output.backpropagate(net.hidden.value, error, rate)
    net.hidden.backpropagate(input, residual, rate)
}

func (net *Network) String() string {
    return fmt.Sprintf(
        "Hidden=[Weights=%v, Bias=%v]\n" +
        "Output=[Weights=%v, Bias=%v]",
        net.hidden.weight, net.hidden.bias,
        net.output.weight, net.output.bias)
}

func MeanSquaredError (result []float64, expected []float64) float64 {
    sum := 0.0
    for i := 0; i < len(result); i++ {
        sum += math.Pow(expected[i] - result[i], 2)
    }
    return sum / float64(len(result))
}

func SeedRandom () {
    rand.Seed(time.Nanoseconds())
}
