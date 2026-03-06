CUDA_HOME  ?= /usr/local/cuda
NCCL_HOME  ?= /usr/local/cuda
MPI_HOME   ?= /opt/amazon/openmpi

MPICC      = $(MPI_HOME)/bin/mpicc

CFLAGS     = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LDFLAGS    = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -Wl,-rpath,$(CUDA_HOME)/lib64
LIBS       = -lcudart -lnccl

TARGET     = nccl_sendrecv_test

all: $(TARGET)

$(TARGET): nccl_sendrecv_test.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
