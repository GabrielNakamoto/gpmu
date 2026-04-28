from parser import ptx_to_ir
from runtime import SIMTCore, Dim3

src = open("test/gemm.ptx").read()
kernels = ptx_to_ir(src)

k = next(iter(kernels.values()))
print("Kernel name:", k.name)
print("Params:", k.params)
print("Regs:", k.regs)
print("Ops:", [op.opcode for op in k.instructions])

sm = SIMTCore()
sm(Dim3(1), Dim3(256), k, 1, 2, 3, 4, 5, 6)
