package main
import (
    "neural"
    "encoding/binary"
    "io"
    "os"
    "fmt"
    "flag"
)

func ReadMNISTLabels (r io.Reader) (labels []byte) {
    header := [2]int32{}
    binary.Read(r, binary.BigEndian, &header)
    labels = make([]byte, header[1])
    r.Read(labels)
    return
}

func ReadMNISTImages (r io.Reader) (images [][]byte, width, height int) {
    header := [4]int32{}
    binary.Read(r, binary.BigEndian, &header)
    images = make([][]byte, header[1])
    width, height = int(header[2]), int(header[3])
    for i := 0; i < len(images); i++ {
        images[i] = make([]byte, width * height)
        r.Read(images[i])
    }
    return
}

func ImageString (buffer []byte, height, width int) (out string) {
    for i, y := 0, 0; y < height; y++ {
        for x := 0; x < width; x++ {
            if buffer[i] > 128 { out += "#" } else { out += " " }
            i++
        }
        out += "\n"
    }
    return
}

func OpenFile (path string) *os.File {
    file, err := os.Open(path)
    if (err != nil) {
        fmt.Println(err)
        os.Exit(-1)
    }
    return file
}

func main () {
    const numLabels = 10
    const epsilon = 0.001
    const hiddenNodes = 100
    const pixelRange = 255
    const learningRate = 0.75
    const momentum = 0.25

    labelFile := flag.String("l", nil, "label file")
    imageFile := flag.String("i", nil, "image file")
    dumpFile  := flag.String("d", "mnist.json", "dump file")

    if labelFile == nil || imageFile == nil {
        flag.Usage()
        os.Exit(-2)
    }

    labelData := ReadMNISTLabels(OpenFile(labelFile))
    imageData, width, height := ReadMNISTImages(OpenFile(imageFile))

    labels := make([][]float64, len(labelData))
    for i, value := range labelData {
        labels[i] = make([]float64, numLabels)
        labels[i][value] = 1.0
    }

    images := make([][]float64, len(imageData))
    for i, vector := range imageData { 
        images[i] = make([]float64, len(vector))
        for j := 0; j < len(images[i]); j++ {
            images[i][j] = float64(vector[j])/pixelRange
        }
    }

    var net *neural.Network
    if file, err := os.Open(dumpFile); err != nil {
        neural.SeedRandom()
        net = neural.NewNetwork(width * height, hiddenNodes, numLabels)
    } else {
        net = neural.LoadNetwork(file)
    }

    epoch, best := 0, epsilon;
    for ; best >= epsilon; epoch++ {
        best = 0.0
        for i, expected := range labels {
            input := images[i]
            result := net.Activate(input)
            net.Train(input, expected, learningRate, momentum)
            err := neural.MeanSquaredError(result, expected)
            if err > best { best = err }
            if i % 1000 == 0 {
                fmt.Printf("\rEpoch # %d: %d%%", epoch, int(float32(i)/float32(len(labels))*100.0))
            }
        }
        fmt.Println("\rEpoch #", epoch, "@ MSE =", best)
        file, _ := os.Create(dumpFile)
        net.Save(file)
    }
}
