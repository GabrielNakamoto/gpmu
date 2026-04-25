from ptx import Kernel
from runtime import SM, Dim3

kernels = Kernel.load_ptx_kernels(open("test/gemm.ptx", 'r').read())
gpu = SM(next(iter(kernels.values())), Dim3(4, 4), Dim3(16, 16))
gpu(0, [1, 2, 3])
