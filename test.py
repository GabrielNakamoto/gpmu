from gpmu import SIMTCore, Dim3, load_kernels

kernels = load_kernels(open("test/gemm.ptx", 'r').read())
gpu = SIMTCore()

gpu(next(iter(kernels.values())), Dim3(1), Dim3(256), 1, 2, 3, 4, 5, 6)
