package main
import (
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

func ReadMNISTImages (r io.Reader) (images [][]byte) {
    header := [4]int32{}
    binary.Read(r, binary.BigEndian, &header)
    images = make([][]byte, header[1])
    imageSize := header[2] * header[3]
    for i := 0; i < len(images); i++ {
        image := make([]byte, imageSize)
        r.Read(image)
        images[i] = image
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
    labels := ReadMNISTLabels(OpenFile(os.Args[1]))
    images := ReadMNISTImages(OpenFile(os.Args[2]))
    fmt.Println("Labels =", len(labels), "Images =", len(images), "Image Size =", len(images[0]))
}
