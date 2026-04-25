from __future__ import annotations
from ptx import Instruction, Register, Kernel
import math
import itertools
from dataclasses import dataclass

@dataclass
class Dim3:
    x: int
    y: int = 1
    z: int = 1


class Warp:
    def __init__(self, bid, parent: SM, wid):
        self.pc = 0
        self.bid = bid
        tx, ty = parent.threaddims.x, parent.threaddims.y
        # (lane-local registers, thread idx)
        self.lanes: list[tuple[dict[str, Register], Dim3]]  = [
            (parent.kernel.regfile.copy(), Dim3(lid % tx, int((lid / tx) % ty), int(lid / (tx * ty))))
            for lid in range(wid*32, (wid*32)+32, 1)
        ]
        self.lane_mask = 0xFFFFFFFF
    # execute instruction across lanes, handle divergence
    def __call__(self, op: Instruction):
        print(op.opcode)

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
            warps = [Warp(Dim3(bz, bx, by), self, i) for i in range(self.warps_per_block)]
            if self.warp_remainder > 0: warps[-1].lane_mask = (1 << self.warp_remainder) - 1

            l1_cache = bytearray(48000)
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
