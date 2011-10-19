Neural Go
-=-=-=-=-

This package implements a very simple multilayer perceptron network in Go, with
gradient descent training via backpropagation. The library builds with `make
all`.

Included is a simple example to train a 6 node network on the XOR function. Run
`make xor` to build it. Of course, the example doesn't output anything really
impressive, just iterates until the mean squared error of the output is below a
certain threshold for all training examples. Doesn't take long at all on my
laptop, though.

The other example is `mnist.go`, which is designed to train on the MNIST
handwritten digits dataset. I've got it up to over 96% accuracy in a few
epochs, like so:

    $ wget -r -np -Agz http://yann.lecun.com/exdb/mnist/
    $ mv yann.lecun/exdb/mnist/*.gz .
    $ for i in *gz; do gunzip $i; done
    $ make mnist
    $ ./mnist -si train-images-idx3-ubyte \
              -sl train-labels-idx1-ubyte \
              -ti t10k-images-idx3-ubyte \
              -tl t10k-labels-idx1-ubyte

I'm sure this codebase could easily be made to do better. See
http://yann.lecun.com/exdb/mnist/ for more details on the dataset.

The `niceidea.go` file contains a sketch of a neural network parallelized in
goroutines. It doesn't work. And my attempt to parallelize `neural.go`
resulted in a version that was ~4x slower, because of the cost of using
channels to synchronize the activation and backpropagation steps. Oh well. That
was interesting, anyway.

Mostly, I did this to experiment with building things in Go, and because I'd
never actually successfully implemented backpropagation before. This code is
Public Domain; do what you like with it. It is not guaranteed to work or to be
useful for any purpose. Patches welcome!

SDE
2011/10/16
San Francisco, CA
