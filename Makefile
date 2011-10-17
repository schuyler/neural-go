include $(GOROOT)/src/Make.inc

TARG=neural
GOFILES=neural.go

include $(GOROOT)/src/Make.pkg

xor: _obj/neural.a xor.go
	$(GC) -I _obj xor.go
	$(LD) -L _obj -o xor xor.$(O)

realclean: clean
	rm -f xor
