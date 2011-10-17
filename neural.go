package neural
import (
    "math"
    "rand"
    "time"
    "fmt"
    "json"
    "io"
)

type Layer struct {
    Weight [][]float64
    Bias []float64
    delta [][]float64
    value []float64
}

type Network struct {
    Hidden *Layer 
    Output *Layer
}

func randomWeight () float64 {
    return rand.Float64() * 2.0 - 1.0
}

func (layer *Layer) initialize () {
    layer.value = make([]float64, len(layer.Weight))
    layer.delta = make([][]float64, len(layer.Weight))
    for i := 0; i < len(layer.delta); i++ {
        layer.delta[i] = make([]float64, len(layer.Weight[0]))
    }
}

func newLayer(inputs int, nodes int) (layer *Layer) {
    layer = new(Layer)
    layer.Weight = make([][]float64, nodes)
    for i := 0; i < nodes; i++ {
        layer.Weight[i] = make([]float64, inputs)
        for j := 0; j < inputs; j++ {
            layer.Weight[i][j] = randomWeight()
        }
    }
    layer.Bias = make([]float64, nodes)
    for i := 0; i < nodes; i++ {
        layer.Bias[i] = randomWeight()
    }
    layer.initialize()
    return
}

func NewNetwork(inputs int, hiddens int, outputs int) (net *Network) {
    net = new(Network)
    net.Hidden = newLayer(inputs, hiddens)
    net.Output = newLayer(hiddens, outputs)
    return
}

func (layer *Layer) feedforward(input []float64) []float64 {
    for i := 0; i < len(layer.value); i++ {
        sum := layer.Bias[i]
        for j := 0; j < len(input); j++ {
            sum += layer.Weight[i][j] * input[j]
        }
        layer.value[i] = 1.0 / (1.0 + math.Pow(math.E, -sum))
    }
    return layer.value
}

func (net *Network) Activate(input []float64) (result []float64) {
    hidden := net.Hidden.feedforward(input)
    output := net.Output.feedforward(hidden)
    result = make([]float64, len(output))
    copy(result, output)
    return
}

func (layer *Layer) backpropagate (input []float64, error []float64, rate float64, accel float64) (residual []float64) {
    residual = make([]float64, len(layer.Weight[0]))
    for i, weight := range layer.Weight {
        delta := error[i] * layer.value[i] * (1.0 - layer.value[i])
        for j := 0; j < len(weight); j++ {
            residual[j] += delta * weight[j]
            layer.delta[i][j] = rate * delta * input[j] + accel * layer.delta[i][j]
            weight[j] += layer.delta[i][j]
        }
        layer.Bias[i] += rate * delta 
    }
    return
}

func (net *Network) Train(input []float64, expected []float64, rate float64, accel float64) {
    error := make([]float64, len(net.Output.value))
    for i := 0; i < len(error); i++ {
        error[i] = expected[i] - net.Output.value[i]
    }
    residual := net.Output.backpropagate(net.Hidden.value, error, rate, accel)
    net.Hidden.backpropagate(input, residual, rate, accel)
}

func (net *Network) String() string {
    return fmt.Sprintf(
        "Hidden=[Weights=%v, Bias=%v]\n" +
        "Output=[Weights=%v, Bias=%v]",
        net.Hidden.Weight, net.Hidden.Bias,
        net.Output.Weight, net.Output.Bias)
}

func (net *Network) Save(w io.Writer) {
    enc := json.NewEncoder(w)
    enc.Encode(net)
}

func LoadNetwork(r io.Reader) *Network {
    net := new(Network)
    dec := json.NewDecoder(r)
    dec.Decode(net)
    net.Hidden.initialize()
    net.Output.initialize()
    return net
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
