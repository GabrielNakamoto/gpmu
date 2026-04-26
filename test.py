from gpmu import load_kernels

kernels = load_kernels(open("test/kernel.ptx", 'r').read())
