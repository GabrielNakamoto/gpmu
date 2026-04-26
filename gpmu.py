from __future__ import annotations
import struct, re, itertools, math
from dataclasses import dataclass
from enum import Enum, auto

GLOBAL_MEMORY_SIZE = 48000
SHARED_MEMORY_SIZE = 48000
RO_MEMORY_SIZE = 48000

fmtmap = {
    "f16" : "e", "f32" : "f",
    "s8" : "b", "s16" : "h", "s32" : "i",
    "u8" : "B", "u16" : "H", "u32" : "I",
    "s64" : "q", "u64" : "Q"
}

def ptx_fmt_bytes(s):
    size = re.search(r'\d+', s)
    return int(size.group()) // 8 if size else 1

@dataclass
class Dim3:
    x: int
    y: int = 1
    z: int = 1

@dataclass
class Instruction:
    opcode: str
    qualifiers: list[str]
    operands: list[str]
    predicate: str | None = None

class StateSpace(Enum):
    reg = auto(); sreg = auto();
    glob = auto(); local = auto();
    const = auto(); param = auto()
    shared = auto();

@dataclass
class WarpKey:
    space: StateSpace
    format: str
    identifier: str

# ---- Runtime/Hardware Classes ----
@dataclass
class WarpState:
    global_mem: bytearray
    shared_mem: bytearray
    const_mem: bytearray
    ro_mem: bytearray

    params: dict[str, bytes]
    regs: dict[str, list[int]]
    sregs: dict[str, list[int]]

    symbol_table: dict[str, int]

    @property
    def space_map(self):
        return {
            "shared" : self.shared_mem,
            "global" : self.global_mem,
            "param" : self.const_mem,
            "const" : self.const_mem
        }

    def resolve_addr(self, expr, lane) -> int:
        expr = expr[1:-1]
        base, offset = 0, 0
        if len(expr.split('+')) == 2:
            expr, n = expr.split('+')
            offset = int(n.strip())
        if expr in self.regs: base = self.regs[expr][lane]
        elif expr in self.sregs: base = self.sregs[expr][lane]
        elif expr in self.symbol_table: base = self.symbol_table[expr]
        return int(base) + offset

    def space_view(self, space, fmt, address, lane):
        ptr = self.resolve_addr(address, lane)
        nbytes = ptx_fmt_bytes(fmt)
        buffer = self.space_map.get(space, None)
        if not buffer: raise Exception
        return memoryview(buffer)[ptr:ptr+nbytes]

    def get(self, space, fmt, address, lane):
        got = struct.unpack(fmtmap[fmt], self.space_view(space, fmt, address, lane))[0]
        print("Got:", got)
        return got

    def set(self, space, fmt, address, lane, value):
        view = self.space_view(space, fmt, address, lane)
        print("Set:", value)
        view[:] = struct.pack(fmtmap[fmt], value)

class Warp:
    def __init__(self, stream: list[Instruction], bdims: Dim3, bid: Dim3, warp_id: int, state: WarpState, lane_mask=0xFFFFFFFF):
        self.stream = stream
        self.lane_mask = lane_mask
        self.state = state
        self.pc = 0

        # fill special registers
        tid = Dim3(1)
        self.state.sregs["%tid.x"] = [tid.x for _ in range(32)]
        self.state.sregs["%tid.y"] = [tid.x for _ in range(32)]
        self.state.sregs["%tid.z"] = [tid.x for _ in range(32)]
        self.state.sregs["%ntid.x"] = [bdims.x for _ in range(32)]
        self.state.sregs["%ntid.y"] = [bdims.y for _ in range(32)]
        self.state.sregs["%ntid.z"] = [bdims.z for _ in range(32)]
    def __call__(self):
        op = self.stream[self.pc]
        # print(op)

        def get_value(s, lane):
            s = s.strip()
            if s[0]=="%": return self.state.regs[s][lane] if s in self.state.regs else self.state.sregs[s][lane]
            try: return float(s)
            except ValueError: pass
            return int(s)
        arithmap = {
            "add" : lambda a, b : a + b,
            "sub" : lambda a, b : a - b,
            "mul" : lambda a, b : a * b,
            "div" : lambda a, b : a / b
        }

        print(op)
        for lane in range(32):
            if not ((self.lane_mask >> lane) & 1): continue
            if op.opcode in ["add", "sub", "mul", "div"]:
                d, a, b = op.operands
                a, b = get_value(a, lane), get_value(b, lane)
                self.state.regs[d][lane]=arithmap[op.opcode](a, b)
            elif op.opcode == "ld":
                d, addr = op.operands
                space, fmt = op.qualifiers
                self.state.regs[d][lane]=self.state.get(space, fmt, addr, lane)
            elif op.opcode == "st":
                addr, a = op.operands
                space, fmt = op.qualifiers
            elif op.opcode == "mov":
                fmt = op.qualifiers
                d, a = op.operands

        self.pc += 1
        # print(self.state.regs)
        pass

class WarpScheduler:
    def __init__(self, warps: list[Warp]):
        self.warps = warps
    def __iter__(self): return self
    def __next__(self):
        return self.warps[0]
        # raise StopIteration

class SIMTCore:
    def __init__(self):
        self.global_memory = bytearray(GLOBAL_MEMORY_SIZE)
    def __call__(self, kernel: Kernel, gdims: Dim3, bdims: Dim3, *hostparams):
        threads_per_block = bdims.x * bdims.y * bdims.z
        warps_per_block = math.ceil(threads_per_block / 32)
        active_last_lanes = threads_per_block % 32

        regfile  = {k : [0] * 32 for k, _ in kernel.registers}
        shared_mem, ro_mem, const_mem = bytearray(SHARED_MEMORY_SIZE), bytearray(RO_MEMORY_SIZE), bytearray(24000)

        for i, (fmt, name) in enumerate(kernel.params):
            const_mem[kernel.symbol_table[name]:]=struct.pack(fmtmap[fmt[1:]], hostparams[i])

        for bz, by, bx in itertools.product(range(bdims.z), range(bdims.y), range(bdims.x)):
            warps = [
                Warp(kernel.instructions, bdims, Dim3(bz, by, bx), wid,
                     WarpState(self.global_memory, shared_mem, const_mem, ro_mem, dict(), regfile, dict(), kernel.symbol_table))
                for wid in range(warps_per_block)
            ]
            if active_last_lanes > 0: warps[-1].lane_mask = (1<<active_last_lanes)-1
            schedulers = [WarpScheduler(warps[i::4]) for i in range(4)]
            active = True
            while active:
                active = False
                for ws in schedulers:
                    if warp := next(ws):
                        active = True
                        warp()

# ---- Kernel Program parsed from .ptx assembly ----
@dataclass
class Kernel:
    params: list[tuple[str, str]]
    registers: list[tuple[str, str]]
    symbol_table: dict[str, int]
    instructions: list[Instruction]
    label_map: dict[str, int]

def load_kernels(src: str) -> dict[str, Kernel]:
    # ignore everything but kernel entries
    kernels = dict()
    for match in re.finditer(r'(?:\.\w+\s+)*\.entry\s+(\S+)\s*\(([^)]*)\)\s*\{(.*?)\}', src, re.S):
        name, param_str, body_str = match.group(1), match.group(2), match.group(3)
        params = re.findall(r'\.(param\s\S+\s[^,\s]+)', param_str)

        raw_lines = [l.strip() for l in body_str.split(';') if l.strip()]
        directives, lines, label_map = params.copy(), [], {}
        for l in raw_lines:
            if ':' in l:
                label, rest = l.split(':')
                label_map[label.strip()]=len(lines)
                l = rest.strip()
            if not l: continue
            if l[0]=='.': directives.append(l[1:])
            else: lines.append(l)

        shared_offset, global_offset, const_offset = 0, 0, 0
        symbol_table = dict()
        registers = []
        for d in directives:
            if d.split()[0].strip() in StateSpace.__members__.keys():
                space, format, identifier = d.split()
                nbytes = ptx_fmt_bytes(format)
                if space == "shared":
                    symbol_table[identifier]=shared_offset
                    shared_offset += nbytes
                elif space == "global":
                    symbol_table[identifier]=global_offset
                    global_offset += nbytes
                elif space == "param" or space == "const":
                    symbol_table[identifier]=const_offset
                    const_offset += nbytes
                elif space == "reg":
                    m = re.search(r'(\%[^<]+)\<([^>]+)\>', identifier)
                    if not m: continue
                    for i in range(int(m.group(2))): registers.append((f"{m.group(1)}{i}", format))

        # TODO: predicate handling
        ops = []
        for line in lines:
            match = re.match(r'(\S+)\s*(.*)', line)
            if not match: continue
            opstrs, operands = match.group(1).split('.'), [o.strip() for o in match.group(2).split(',')]
            opcode, qualifiers = opstrs[0], opstrs[1:]
            ops.append(Instruction(opcode, qualifiers,operands))
        kernels[name]=Kernel([s.split()[1:] for s in params], registers, symbol_table, ops, dict())
    return kernels
