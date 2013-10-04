package neural

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"time"
)

type Float float32

type Layer struct {
	Weight [][]Float
	Bias   []Float
	delta  [][]Float
	value  []Float
}

type Network struct {
	Hidden *Layer
	Output *Layer
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

func randomWeight() Float {
	return Float(rand.Float64()*2.0 - 1.0)
}

func (layer *Layer) initialize() {
	layer.value = make([]Float, len(layer.Weight))
	layer.delta = make([][]Float, len(layer.Weight))
	for i := 0; i < len(layer.delta); i++ {
		layer.delta[i] = make([]Float, len(layer.Weight[0]))
	}
}

func newLayer(inputs int, nodes int) (layer *Layer) {
	layer = new(Layer)
	layer.Weight = make([][]Float, nodes)
	for i := 0; i < nodes; i++ {
		layer.Weight[i] = make([]Float, inputs)
		for j := 0; j < inputs; j++ {
			layer.Weight[i][j] = randomWeight()
		}
	}
	layer.Bias = make([]Float, nodes)
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

func (layer *Layer) feedforward(input []Float) []Float {
	for i := 0; i < len(layer.value); i++ {
		sum := layer.Bias[i]
		for j := 0; j < len(input); j++ {
			sum += layer.Weight[i][j] * input[j]
		}
		layer.value[i] = Float(1.0 / (1.0 + math.Pow(math.E, -float64(sum))))
	}
	return layer.value
}

func (net *Network) Activate(input []Float) (result []Float) {
	hidden := net.Hidden.feedforward(input)
	output := net.Output.feedforward(hidden)
	result = make([]Float, len(output))
	copy(result, output)
	return
}

func (layer *Layer) backpropagate(input []Float, error []Float, rate Float, accel Float) (residual []Float) {
	residual = make([]Float, len(layer.Weight[0]))
	for i, weight := range layer.Weight {
		delta := error[i] * layer.value[i] * (1.0 - layer.value[i])
		for j := 0; j < len(weight); j++ {
			residual[j] += delta * weight[j]
			layer.delta[i][j] = rate*delta*input[j] + accel*layer.delta[i][j]
			weight[j] += layer.delta[i][j]
		}
		layer.Bias[i] += rate * delta
	}
	return
}

func (net *Network) Train(input []Float, expected []Float, rate Float, accel Float) {
	error := make([]Float, len(net.Output.value))
	for i := 0; i < len(error); i++ {
		error[i] = expected[i] - net.Output.value[i]
	}
	residual := net.Output.backpropagate(net.Hidden.value, error, rate, accel)
	net.Hidden.backpropagate(input, residual, rate, accel)
}

func (net *Network) String() string {
	return fmt.Sprintf(
		"Hidden=[Weights=%v, Bias=%v]\n"+
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

func MeanSquaredError(result []Float, expected []Float) Float {
	sum := 0.0
	for i := 0; i < len(result); i++ {
		sum += math.Pow(float64(expected[i]-result[i]), 2)
	}
	return Float(sum) / Float(len(result))
}
