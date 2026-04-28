from parser import Kernel, Instruction
import re, struct, math, itertools
from collections import Counter
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
    regs: list[dict[str, tuple[str, bytes]]]
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

ptx_to_struct_fmt = {
    "pred" : "?",
    "s8" : "b", "s16" : "h", "s32" : "i", "s64" : "q",
    "u8" : "B", "u16" : "H", "u32" : "I", "u64" : "Q",
    "f16" : "e", "f32" : "f", "f64" : "d",
}

def ptx_width(fmt): return int(re.search(r'\d+', fmt).group()) // 8

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

    def assign_reg(self, lane, op_fmt, key, value):
        decl_fmt = self.state.regs[lane][key][0]
        width = ptx_width(decl_fmt)
        if isinstance(value, (bytes, bytearray)):
            raw = bytes(value).ljust(width, b"\x00")[:width]
        elif decl_fmt.startswith("f") or op_fmt.startswith("f"):
            raw = struct.pack("<" + ptx_to_struct_fmt[op_fmt], float(value))
            raw = raw.ljust(width, b"\x00")[:width]
        else:
            mask = (1 << (width * 8)) - 1
            raw = (int(value) & mask).to_bytes(width, "little", signed=False)
        self.state.regs[lane][key]=(decl_fmt, raw)

    def get_value(self, lane, op_fmt, opr):
        if opr[0] != "reg": return opr[1]
        raw = self.state.regs[lane][opr[1]][1]
        buf = raw[:ptx_width(op_fmt)]
        if op_fmt.startswith("f"):
            return struct.unpack("<" + ptx_to_struct_fmt[op_fmt], buf)[0]
        return int.from_bytes(buf, "little", signed=op_fmt.startswith("s"))

    def resolve_address(self, addr: str):
        base, offset = 0, 0
        if "+" in addr:
            addr, n = addr.split("+")
            offset = int(n)
        if addr.startswith("%"):
            pass
        return base + offset

    def get_from_address(self, lane, space, op_fmt, addr):
        pass

    def __call__(self):
        active = [l for l in range(32) if self.alive[l]]
        optimal_pc = max(Counter([self.pc[l] for l in active]))
        issue_mask = [l for l in active if self.pc[l] == optimal_pc]
        op = self.stream[optimal_pc]

        for l in issue_mask:
            if op.opcode in arith_ops:
                d, fmt, args = op.operands[0], op.qualifiers[-1], op.operands[1:]
                self.assign_reg(l, fmt, d[1], arith_ops[op.opcode](*[self.get_value(l, fmt, a) for a in args]))
            elif op.opcode == "ld":
                d, addr = op.operands
                space, fmt = op.qualifiers[0], op.qualifiers[-1]
                print(op.opcode, addr)
            elif op.opcode == "st":
                pass
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
        for (fmt, prefix, n) in kernel.regs:
            for i in range(n):
                raw=struct.pack(ptx_to_struct_fmt[fmt], 0) if not fmt.startswith("b") else int(0).to_bytes(ptx_width(fmt), "little", signed=False)
                regs[f"{prefix}{i}"]=(fmt, raw)
        regs = [regs.copy() for _ in range(32)]

        for (gz, gy, gx) in itertools.product(range(grid.z), range(grid.y), range(grid.x)):
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
