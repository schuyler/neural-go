package main
import (
    "neural"
    "encoding/binary"
    "io"
    "os"
    "fmt"
    "flag"
    "runtime/pprof"
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
    const learningRate = 0.25
    const momentum = 0

    labelFile := flag.String("l", "", "label file")
    imageFile := flag.String("i", "", "image file")
    dumpFile  := flag.String("d", "mnist.json", "dump file")
    memProfile := flag.String("m", "", "memory profile")
    flag.Parse()

    if *labelFile == "" || *imageFile == "" {
        flag.Usage()
        os.Exit(-2)
    }

    fmt.Println("Loading image data...")
    labelData := ReadMNISTLabels(OpenFile(*labelFile))
    imageData, width, height := ReadMNISTImages(OpenFile(*imageFile))

    var net *neural.Network
    if file, err := os.Open(*dumpFile); err != nil {
        neural.SeedRandom()
        net = neural.NewNetwork(width * height, hiddenNodes, numLabels)
    } else {
        fmt.Println("Loading network...")
        net = neural.LoadNetwork(file)
    }

    if *memProfile != "" {
        f, _ := os.Create(*memProfile)
        pprof.WriteHeapProfile(f)
        f.Close()
        return
    }

    input := make([]float64, width * height)
    expected := make([]float64, numLabels)

    epoch, worst, overall := 0, epsilon, 0.0
    for ; worst >= epsilon; epoch++ {
        worst, overall = 0.0, 0.0
        for i, labelIndex := range labelData {
            for j := 0; j < len(input); j++ {
                input[j] = float64(imageData[i][j])/pixelRange * 0.9 + 0.1
            }
            for j := 0; j < len(expected); j++ {
                expected[j] = 0.1
                if j == int(labelIndex) { expected[j] = 0.9 }
            }
            result := net.Activate(input)
            net.Train(input, expected, learningRate, momentum)
            err := neural.MeanSquaredError(result, expected)
            if err > worst { worst = err }
            overall += err
            if i % 1000 == 0 {
                pctDone := int(float32(i)/float32(len(labelData))*100.0)
                fmt.Printf("\rEpoch #%d: %d%%, MSE = %.5f, worst = %.5f", epoch, pctDone, overall/float64(i), worst)
            }
        }
        fmt.Printf("\rEpoch #%d: done, MSE = %.5f, worst = %.5f\n", epoch, overall/float64(len(labelData)), worst)
        file, _ := os.Create(*dumpFile)
        net.Save(file)
    }
}
