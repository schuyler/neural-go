include $(GOROOT)/src/Make.inc

TARG=neural
GOFILES=neural.go

include $(GOROOT)/src/Make.pkg

examples: xor mnist

xor: _obj/neural.a xor.go
	$(GC) -I _obj xor.go
	$(LD) -L _obj -o xor xor.$(O)

mnist: _obj/neural.a mnist.go
	$(GC) -I _obj mnist.go
	$(LD) -L _obj -o mnist mnist.$(O)

realclean: clean
	rm -f xor mnist
