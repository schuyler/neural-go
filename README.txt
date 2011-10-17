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

Mostly, I did this to experiment with building things in Go, and because I'd
never actually successfully implemented backpropagation before.

The `niceidea.go` file contains a sketch of a neural network parallelized in
goroutines. It doesn't work. And my attempt to parallelize `neural.go`
resulted in a version that was ~4x slower, because of the cost of using
channels to synchronize the activation and backpropagation steps. Oh well. That
was interesting, anyway.

This code is Public Domain; do what you like with it. It is not guaranteed to
work or to be useful for any purpose. Patches welcome!

SDE
2011/10/16
San Francisco, CA
