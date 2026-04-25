from ptx import Kernel
from runtime import SM, Dim3

kernels = Kernel.load_ptx_kernels(open("test/kernel.ptx", 'r').read())
gpu = SM(next(iter(kernels.values())), Dim3(1), Dim3(256))
gpu(0, [1, 2, 3])
