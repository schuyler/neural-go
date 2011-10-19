package main
import (
    "neural"
    "encoding/binary"
    "io"
    "os"
    "fmt"
    "flag"
)

const numLabels = 10
const epsilon = 0.001
const hiddenNodes = 100
const pixelRange = 255
const learningRate = 0.25
const momentum = 0.10

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

func pixelWeight (px byte) neural.Float {
   return neural.Float(px) / pixelRange * 0.9 + 0.1 
}

func main () {
    sourceLabelFile := flag.String("sl", "", "source label file")
    sourceImageFile := flag.String("si", "", "source image file")
    testLabelFile := flag.String("tl", "", "test label file")
    testImageFile := flag.String("ti", "", "test image file")
    dumpFile  := flag.String("d", "mnist.json", "dump file")
    numSamples := flag.Int("n", -1, "number of samples (default=all)")
    flag.Parse()

    if *sourceLabelFile == "" || *sourceImageFile == "" {
        flag.Usage()
        os.Exit(-2)
    }

    fmt.Println("Loading training data...")
    labelData := ReadMNISTLabels(OpenFile(*sourceLabelFile))
    imageData, width, height := ReadMNISTImages(OpenFile(*sourceImageFile))

    var testLabelData []byte
    var testImageData [][]byte
    if *testLabelFile != "" && *testImageFile != "" {
        fmt.Println("Loading test data...")
        testLabelData = ReadMNISTLabels(OpenFile(*testLabelFile))
        testImageData, _, _ = ReadMNISTImages(OpenFile(*testImageFile))
    }

    var net *neural.Network
    if file, err := os.Open(*dumpFile); err != nil {
        fmt.Println("Creating network...")
        neural.SeedRandom()
        net = neural.NewNetwork(width * height, hiddenNodes, numLabels)
    } else {
        fmt.Println("Loading network...")
        net = neural.LoadNetwork(file)
    }

    input := make([]neural.Float, width * height)
    expected := make([]neural.Float, numLabels)

    epoch, worst, overall := 0, neural.Float(epsilon), neural.Float(0.0)
    for ; worst >= epsilon; epoch++ {
        worst, overall = 0.0, 0.0
        for i, labelIndex := range labelData {
            for j := 0; j < len(input); j++ {
                input[j] = pixelWeight(imageData[i][j])
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
            if i % int(len(labelData)/100) == 0 {
                pctDone := int(float32(i)/float32(len(labelData))*100.0)
                if *numSamples > 0 {
                    pctDone = int(float32(i)/float32(*numSamples)*100.0)
                }
                fmt.Printf("\rEpoch #%d: %d%%, MSE = %.5f, worst = %.5f", epoch, pctDone, overall/neural.Float(i), worst)
            }
            if *numSamples > 0 && *numSamples == i {
                break
            }
        }

        correct, total := 0, len(testLabelData)
        if total > 0 {
            for i, labelIndex := range testLabelData {
                for j := 0; j < len(input); j++ {
                    input[j] = pixelWeight(testImageData[i][j])
                }
                result := net.Activate(input)
                selected, maxValue := 0, neural.Float(-1.0)
                for j, value := range result {
                    if value >= maxValue {
                        selected = j
                        maxValue = value
                    }
                }
                if selected == int(labelIndex) {
                    correct += 1
                }
                if *numSamples > 0 && *numSamples == i {
                    break
                }
            }
        }

        fmt.Printf("\rEpoch #%d: done, MSE = %.5f, worst = %.5f", epoch, overall/neural.Float(len(labelData)), worst)
        if total > 0 {
            fmt.Printf(", correct = %.2f%%", float32(correct)/float32(total)*100.0)
        }
        fmt.Printf("\n")

        file, _ := os.Create(*dumpFile)
        net.Save(file)
    }
}
