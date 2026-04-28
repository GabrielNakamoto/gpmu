from parser import Kernel, Instruction
import struct
from collections import Counter
import math
from dataclasses import dataclass

@dataclass
class Dim3:
    x: int
    y: int = 1
    z: int = 1

class CacheModel:
    pass

@dataclass
class WarpState:
    global_mem: bytearray
    shared_mem: bytearray
    const_mem: bytearray
    regs: list[dict[str, bytes]]
    sregs: list[dict[str, int]]

arith_ops = {
    "add" : lambda a, b : a + b,
    "sub" : lambda a, b : a - b,
    "mul" : lambda a, b : a * b,
    "div" : lambda a, b : a / b,
    "fma" : lambda a, b, c : a + b*c,
    "neg" : lambda a : -1 * a,
    "rem" : lambda a, b : a % b,
    "abs" : abs,
    "min" : min,
    "max" : max,
}

class Warp:
    def __init__(self, wid: int, gid: tuple[int, int, int], grid: Dim3, cta: Dim3, state: WarpState, stream: list[Instruction]):
        def filltup(key, tup: Dim3 | tuple[int, int, int]): 
            x, y, z = tup if isinstance(tup, tuple) else (tup.x, tup.y, tup.z)
            for n in range(32):
                state.sregs[n][f"%{key}.x"]=x
                state.sregs[n][f"%{key}.y"]=y
                state.sregs[n][f"%{key}.z"]=z

        tid = (wid % cta.x, int(wid / cta.x) % cta.y, int(wid / (cta.x * cta.y)))

        filltup("ntid", cta)
        filltup("tid", tid)
        filltup("nctaid", grid)
        filltup("ctaid", gid)

        self.stream = stream
        self.state = state
        self.lane_mask = (1 << 32) - 1
        # independent thread scheduling, Volta+ arch
        self.pc = [0 for _ in range(32)]
        self.alive = [True for _ in range(32)]

    def assign_reg(self, lane, fmt, key, value):
        print(fmt)
        self.state.regs[lane][key]=struct.pack(fmt, value)
    def get_value(self, lane, opr):
        if opr[0] == "reg": return self.state.regs[lane][opr[1]]
        else: return opr[1]

    def __call__(self):
        active = [l for l in range(32) if self.alive[l]]
        optimal_pc = max(Counter([self.pc[l] for l in active]))
        issue_mask = [l for l in active if self.pc[l] == optimal_pc]
        op = self.stream[optimal_pc]

        for l in issue_mask:
            if op.opcode in arith_ops:
                fmt = op.qualifiers[0]
                d = op.operands[0]
                args = op.operands[1:]

                self.assign_reg(l, fmt, d[1], arith_ops[op.opcode](*[self.get_value(l, a) for a in args]))
                print(op.opcode, d, args)
        for l in issue_mask: self.pc[l] += 1

class WarpScheduler:
    def __init__(self, warps: list[Warp]):
        self.warps = warps
    def __iter__(self): return self
    def __next__(self):
        return self.warps[0]

class SIMTCore:
    def __call__(self, grid: Dim3, cta: Dim3, kernel: Kernel, *kparams):
        assert len(kparams) == len(kernel.params)
        warps_per_cta = math.ceil(cta.x * cta.y * cta.z / 32)

        global_mem, shared_mem, const_mem = bytearray(48000), bytearray(48000), bytearray(48000)
        regs = {}
        for (_, prefix, n) in kernel.regs:
            for i in range(n): regs[f"{prefix}{i}"]=0
        regs = [regs.copy() for _ in range(32)]

        for gz in range(grid.z):
            for gy in range(grid.y):
                for gx in range(grid.x):
                    warps = [
                        Warp(wid, (gx, gy, gz), grid, cta,
                             WarpState(global_mem, shared_mem, const_mem, regs.copy(), [dict() for _ in range(32)]),
                             kernel.instructions)
                        for wid in range(warps_per_cta)]
                    schedulers = [WarpScheduler(warps[i::4]) for i in range(4)]

                    while True:
                        warps_active = False
                        for ws in schedulers:
                            if warp := next(ws):
                                warps_active = True
                                warp()
                        if not warps_active: break
