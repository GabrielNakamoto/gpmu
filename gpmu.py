from __future__ import annotations
import re
import itertools
import math
from dataclasses import dataclass
from enum import Enum, auto

GLOBAL_MEMORY_SIZE = 48000
SHARED_MEMORY_SIZE = 48000
RO_MEMORY_SIZE = 48000

@dataclass
class Dim3:
    x: int
    y: int = 1
    z: int = 1

class StateSpace(Enum):
    reg = auto(); sreg = auto();
    glob = auto(); local = auto();
    const = auto(); param = auto()

@dataclass
class WarpValue:
    space: StateSpace
    ptx_type: str
    identifier: str

@dataclass
class Instruction:
    opcode: str
    qualifiers: list[str]
    operands: list
    predicate: str | None = None

# ---- Runtime/Hardware Classes ----
@dataclass
class WarpState:
    l2_cache: bytearray
    l1_cache: bytearray
    ro_cache: bytearray
    regs: dict[str, WarpValue]
    sregs: dict[str, WarpValue]

class Warp:
    def __init__(self, bdims: Dim3, bid: Dim3, warp_id: int, state: WarpState):
        self.state = state
        # fill special registers
    def __call__(self, op: Instruction):
        pass

class WarpScheduler:
    def __init__(self, warps: list[Warp]):
        self.warps = warps
        self.pc = 0
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

class SIMTCore:
    def __init__(self):
        self.global_memory = bytearray(GLOBAL_MEMORY_SIZE)
    def __call__(self, kernel: Kernel, gdims: Dim3, bdims: Dim3, *params):
        threads_per_block = bdims.x * bdims.y * bdims.z
        warps_per_block = math.ceil(threads_per_block / 32)
        inactive_last_lanes = threads_per_block % 32

        ro_memory = bytearray(RO_MEMORY_SIZE)
        shared_memory = bytearray(SHARED_MEMORY_SIZE)
        for bz, by, bx in itertools.product(range(bdims.z), range(bdims.y), range(bdims.x)):
            warps = [
                Warp(bdims, Dim3(bz, by, bx), wid,
                     WarpState(self.global_memory, shared_memory, ro_memory, dict(), dict()))
                for wid in range(warps_per_block)
            ]
            schedulers = [WarpScheduler(warps[i::4]) for i in range(4)]
            for ws in schedulers:
                if warp := next(ws):
                    pass

# ---- Kernel Program parsed from .ptx assembly ----
@dataclass
class Kernel:
    params: list[WarpValue]
    registers: list[WarpValue]
    instructions: list[Instruction]
    label_map: dict[str, int]


def load_kernels(src: str) -> dict[str, Kernel]:
    # ignore everything but kernel entries
    kernels = dict()
    for match in re.finditer(r'(?:\.\w+\s+)*\.entry\s+(\S+)\s*\(([^)]*)\)\s*\{(.*?)\}', src, re.S):
        name, param_str, body_str = match.group(1), match.group(2), match.group(3)
        params = [WarpValue(StateSpace.param, ptype, pname)
            for ptype, pname in re.findall(r'\.param\s(\S+)\s([^,\s]+)', param_str)]
        lines = [l.strip() for l in body_str.split(';') if l.strip()]
        directives = [l for l in lines if l[0] == '.']
        instructions = [l for l in lines if l[0] != '.']
        for l in lines: print(l)
    return kernels
