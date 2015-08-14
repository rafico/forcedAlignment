# common flags
CFLAGS=-g -Wall -fPIC -O3

# A good BLAS implementation is required for jsgd. When compiling for
# matlab, we use the implementation shipped with it (-lmwblas, that
# also requires 64-bit integers). Else (for Octave and Python) we use
# the system library. On Linux (Debian 6 and Fedore 14), the Atlas
# implementation is reasonably fast, but must be pointed to explicitly
# because -lblas is not efficient:

BLASLIB=/usr/lib64/atlas/libf77blas.so.3

all : x_matrix.o jsgd.o libjsgd.a

x_matrix.o: x_matrix.c x_matrix.h
	gcc -c -o $@ $< $(CFLAGS) -Dinteger=long -DBLAS_WITH_THREADS -fopenmp

jsgd.o: jsgd.c jsgd.h x_matrix.h
	gcc -c -o $@ $< $(CFLAGS) -fopenmp

test.o : test.cpp
	g++ -c -o $@ $< $(CFLAGS)

libjsgd.a: jsgd.o x_matrix.o
	ar rv $@ $?
	ranlib $@

