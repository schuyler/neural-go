package main
import (
    "neural"
    "fmt"
    "os"
)

func main () {
    neural.SeedRandom()
    const epsilon = 0.001
    training := [][3]neural.Float{{0.1, 0.1, 0.1}, {0.1, 0.9, 0.9}, {0.9, 0.1, 0.9}, {0.9, 0.9, 0.1}}
    net := neural.NewNetwork(2, 3, 1)
    epoch, worst := 0, neural.Float(epsilon);
    for ; worst >= epsilon; epoch++ {
        worst = 0.0
        for _, sample := range training {
            result := net.Activate(sample[0:2])
            net.Train(sample[0:2], sample[2:], 0.75, 0.5)
            err := neural.MeanSquaredError(result, sample[2:])
            if err > worst { worst = err }
        }
        if epoch % 1000 == 0 {
            fmt.Println("Epoch #", epoch, "@ MSE =", worst)
        }
    }
    fmt.Println("Epoch #", epoch, "@ MSE =", worst)
    // fmt.Println(net)
    for _, sample := range training {
        result := net.Activate(sample[0:2])
        fmt.Println("Sample =", sample[0:2],
                 " Expected =", sample[2],
                   " Result =", result[0]) 
    }
    file, _ := os.Create("xor.json")
    net.Save(file)
}
