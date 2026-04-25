from __future__ import annotations
from ptx import Instruction, Register, Kernel, Operand, Reg, Imm, Addr
import math
import itertools
from dataclasses import dataclass

@dataclass
class Dim3:
    x: int
    y: int = 1
    z: int = 1

arithmap = {
    "add": lambda a, b : a + b,
    "sub": lambda a, b : a - b,
    "mul": lambda a, b : a * b,
    "div": lambda a, b : a / b,
}

class WarpState:
    def __init__(
        self,
        l2_cache: bytearray,
        l1_cache: bytearray,
        regfile: dict[str, Register],
        kernel_params
    ):
        self.l2_cache = l2_cache
        self.l1_cache = l1_cache
        self.regs = [regfile.copy() for _ in range(32)]
        self.sregs = [dict() for _ in range(32)]
        self.param_cache = kernel_params
        self.current_lane = 0

    def setlane(self, n): self.current_lane = n

    @property
    def regfile(self): return self.regs[self.current_lane]

    def __getitem__(self, operand) -> int | float:
        if isinstance(operand, Reg): return self.regfile[operand.key].value
        if isinstance(operand, Imm): return operand.value
        if isinstance(operand, Addr): return 0
        return 0

    def set(self, lane: int, space: str, tp: str, dst, src):
        if space == "global":
            pass
        elif space == "param":
            pass
        elif space == "reg":
            pass
        elif space == "sreg":
            pass
        pass

class Warp:
    def __init__(
        self,
        bid: Dim3,
        global_mem: bytearray,
        shared_mem: bytearray,
        kparams,
        regfile: dict[str, Register],
        tdims: Dim3,
        wid: int
    ):
        self.state = WarpState(global_mem, shared_mem, regfile, kparams)
        self.pc = 0
        self.bid = bid

        tx, ty = tdims.x, tdims.y
        for i, lid in enumerate(range(wid*32, (wid*32)+32, 1)):
            tid = Dim3(lid % tx, int((lid / tx) % ty), int(lid / (tx * ty)))
            sregfile = self.state.sregs[i]
            sregfile["tid.x"]=tid.x; sregfile["tid.y"]=tid.y; sregfile["tid.z"]=tid.z;
            sregfile["ntid.x"]=tdims.x; sregfile["ntid.y"]=tdims.y; sregfile["ntid.z"]=tdims.z;

        self.lane_mask = 0xFFFFFFFF

    # execute instruction across lanes, handle divergence
    def __call__(self, op: Instruction):
        for n in range(32):
            self.state.setlane(n)
            if not ((self.lane_mask >> n) & 1): continue
            if op.opcode in arithmap.keys():
                d, a, b = op.operands
                out = arithmap[op.opcode](self.state[n,a], self.state[n,b])
                print(op, "Result:", out)
            elif op.opcode == "ld":
                d, a = op.operands
                space, tp = op.qualifiers
                self.state.set(n, space, tp, d, a)
            elif op.opcode == "mov":
                d, a = op.operands
            elif op.opcode == "st":
                pass

class WarpScheduler:
    def __init__(self, warps: list[Warp]):
        self.warp_pool = warps
        self.pc = 0
    def pick_ready_warp(self, cycle) -> Warp | None:
        return self.warp_pool[0]

class SM:
    def __init__(self, kernel: Kernel, blocks: Dim3, threads: Dim3):
        self.kernel = kernel

        self.griddims = blocks
        self.threaddims = threads
        self.warps_per_block = math.ceil(threads.x * threads.y * threads.z / 32)
        self.warp_remainder = self.warps_per_block % 32
    def __call__(self, *kernel_params):
        global_mem = bytearray(48000)
        for bz, bx, by in itertools.product(range(self.griddims.z), range(self.griddims.y), range(self.griddims.x)):
            l1_cache = bytearray(48000)
            warps = [Warp(Dim3(bz, bx, by), l1_cache, self.kernel.regfile, self.threaddims, i) for i in range(self.warps_per_block)]
            if self.warp_remainder > 0: warps[-1].lane_mask = (1 << self.warp_remainder) - 1

            schedulers = [WarpScheduler(warps[i::4]) for i in range(4)]

            warp_active = True
            cycle = 0
            while warp_active:
                warp_active = False
                for ws in schedulers:
                    warp = ws.pick_ready_warp(cycle)
                    if not warp: continue
                    warp_active = True
                    op = self.kernel.ops[warp.pc]

                    warp(op)
                    warp.pc += 1
                cycle += 1
