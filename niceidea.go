package main
import (
    "rand"
    "math"
    "fmt"
    "strconv"
    //"bufio"
    //"os"
)

type Synapse struct {
    weight float64
    source *Neuron
}

type Neuron struct {
    output chan float64
    inputs []*Synapse
    value float64
    bias float64
    name string
}

type Layer struct {
    nodes []*Neuron
    fixed bool
}

type Network struct {
    input  *Layer
    hidden *Layer
    output *Layer
}

func randomWeight () float64 {
    return rand.Float64() * 2.0 - 1.0
}

func MakeNeuron (name string, in uint, out uint, fixed bool) *Neuron {
    neuron := new(Neuron)
    neuron.output = make(chan float64, out)
    neuron.inputs = make([]*Synapse, 0, in)
    neuron.value = 0.0
    neuron.bias = 0.0
    neuron.name = name
    if ! fixed { neuron.bias = randomWeight() }
    return neuron
}

func MakeLayer (name string, nodes uint, in uint, out uint, fixed bool) *Layer {
    layer := new(Layer)
    layer.fixed = fixed
    layer.nodes = make([]*Neuron, nodes)
    for i := uint(0); i < nodes; i++ {
        layer.nodes[i] = MakeNeuron(name + " #" + strconv.Uitoa(i), in, out, fixed)
    }
    return layer
}

func (neuron *Neuron) String () string {
    return neuron.name
}

func (source *Neuron) connectTo (target *Neuron) {
    synapse := new(Synapse)
    synapse.weight = randomWeight()
    synapse.source = source
    target.inputs = append(target.inputs, synapse)
}

func MakeNetwork (input uint, hidden uint, output uint) *Network {
    net := new(Network)

    net.input = MakeLayer("input", input, 0, hidden, true)
    net.hidden = MakeLayer("hidden", hidden, input, output, false)
    net.output = MakeLayer("output", output, hidden, 1, false)

    for _, inputNode := range net.input.nodes {
        for _, hiddenNode := range net.hidden.nodes {
            inputNode.connectTo(hiddenNode)
        }
    }

    for _, hiddenNode := range net.hidden.nodes {
        for _, outputNode := range net.output.nodes {
            hiddenNode.connectTo(outputNode)
        }
    }

    return net
}

func (neuron *Neuron) activate (value float64) {
    neuron.value = 1.0 / (1 + math.Pow(math.E, -value))
    for i := 0; i < cap(neuron.output); i++ {
        neuron.output <- neuron.value
    }
}

func (neuron *Neuron) feedforward () {
    value := neuron.bias
    for _, synapse := range neuron.inputs {
        value += synapse.weight * <- synapse.source.output
    }
    neuron.activate(value)
}

func (net *Network) run (input []float64) (output []float64) {
    output = make([]float64, len(net.output.nodes))
    for i, neuron := range net.input.nodes {
        go neuron.activate(input[i])
    }
    for _, neuron := range net.hidden.nodes {
        go neuron.feedforward()
    }
    for _, neuron := range net.output.nodes {
        go neuron.feedforward()
    }
    for i, neuron := range net.output.nodes {
        output[i] = <- neuron.output
    }
    return
}


func (neuron *Neuron) backpropagate (rate float64, done chan bool) {
    total := 0.0
    error := 0.0
    for i := 0; i < cap(neuron.output); i++ {
        error += <- neuron.output
    }
    delta := error * neuron.value * (1.0 - neuron.value)
    for _, synapse := range neuron.inputs {
        synapse.weight += rate * delta * synapse.source.value
        total += delta * synapse.weight
    }
    //neuron.bias += rate * delta * neuron.value
    if done == nil {
        for _, synapse := range neuron.inputs {
            synapse.source.output <- total
        }
    } else {
        done <- true
    }
}

func (net *Network) error (expected []float64) float64 {
    sum := 0.0
    for i, neuron := range net.output.nodes {
        sum += math.Pow(expected[i] - neuron.value, 2)
    }
    return sum / float64(len(expected))
}

func (net *Network) train (expected []float64, rate float64) {
    done := make(chan bool, len(net.hidden.nodes))
    for i, neuron := range net.output.nodes {
        neuron.output <- (expected[i] - neuron.value)
        go neuron.backpropagate(rate, nil)
    }
    for _, neuron := range net.hidden.nodes {
        go neuron.backpropagate(rate, done)
    }
    for i := len(net.hidden.nodes); i > 0; i-- { <- done }
}

func main () {
    //in := bufio.NewReader(os.Stdin)
    training := [][3]float64{{0.1, 0.1, 0.1}, {0.1, 0.9, 0.9}, {0.9, 0.1, 0.9}, {0.9, 0.9, 0.1}}
    net := MakeNetwork(2, 3, 1)
    for epoch := 0; true; epoch++ {
        fmt.Println("epoch #", epoch)
        for _, sample := range training {
            result := net.run(sample[:])
            net.train(sample[2:], 0.75)
            /*
            for i, neuron := range net.input.nodes {
                fmt.Println("input", i, neuron)
            }
            for i, neuron := range net.hidden.nodes {
                fmt.Println("hidden", i, neuron)
            }
            */
            err := net.error(sample[2:])
            fmt.Println("sample", sample, "result", result, "error", err)
            //in.ReadLine()
        }
    }
}
