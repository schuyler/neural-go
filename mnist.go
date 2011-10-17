package main
import (
    "neural"
    "encoding/binary"
    "io"
    "os"
    "fmt"
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

    labelData := ReadMNISTLabels(OpenFile(os.Args[1]))
    imageData, width, height := ReadMNISTImages(OpenFile(os.Args[2]))

    labels := make([][]float64, len(labelData))
    for i, value := range labelData {
        labels[i] = make([]float64, numLabels)
        labels[i][value] = 1.0
    }

    images := make([][]float64, len(imageData))
    for i, vector := range imageData { 
        images[i] = make([]float64, len(vector))
        for j := 0; j < len(images[i]); j++ {
            images[i][j] = float64(vector[j])/255.0
        }
    }

    neural.SeedRandom()
    net := neural.NewNetwork(width * height, 100, numLabels)
    epoch, best := 0, epsilon;
    for ; best >= epsilon; epoch++ {
        best = 0.0
        for i, expected := range labels {
            input := images[i]
            result := net.Activate(input)
            net.Train(input, expected, 0.75, 0.25)
            err := neural.MeanSquaredError(result, expected)
            if err > best { best = err }
            if i % 1000 == 0 {
                fmt.Printf("\rEpoch # %d: %d%%", epoch, int(float32(i)/float32(len(labels))*100.0))
            }
        }
        fmt.Println("\rEpoch #", epoch, "@ MSE =", best)
    }
}
