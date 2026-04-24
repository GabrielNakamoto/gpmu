import struct
import re
from dataclasses import dataclass

@dataclass
class Operand:
    pass

@dataclass
class Reg(Operand):
    key: str

@dataclass
class Imm(Operand):
    value: int | float

@dataclass
class Addr(Operand):
    base: str
    offset: int = 0

@dataclass
class Label(Operand):
    id: str

@dataclass
class Instruction:
    opcode: str
    qualifiers: list[str]
    operands: list[Operand]
    predicate: str | None = None

class Register:
    def __init__(self, type_str):
        type_str = type_str[1:] # strip '.'
        sp = next((i for i, x in enumerate(type_str) if str.isdigit(x)), None)
        t = type_str if not sp else type_str[:sp]
        self.signed = t == "s"
        self.pred = type_str == "pred"
        self.fp = t == "f"
        self.bits = 1 if not sp else int(type_str[sp:])
        self.value: int = 0

@dataclass
class Kernel:
    params: dict[str, Register]
    ops: list[Instruction]
    regfile: dict[str, Register]

    @staticmethod
    def decode(param_str, body) -> 'Kernel':
        lines = [line.strip() for line in body.split(';')]
        instrs = [l for l in lines if len(l) > 0 and l[0] != '.']
        dirs = [l for l in lines if len(l) > 0 and l[0] == '.']

        # 0. Strip and register labels
        label_map = {}
        for i, line in enumerate(instrs):
            line = line.strip().split(':')
            if len(line) == 1: continue
            label, rest = line
            label_map[label[1:]] = i
            instrs[i]=rest
        
        # 1. Parse kernel parameters
        params = {}
        for p in param_str.split(','):
            if match := re.search(r'\.param\s+(\S+)\s+(\w+)', p):
                ptype, pname = match.group(1), match.group(2)
                params[pname] = Register(ptype)

        # 2. Fill register file
        registers = {}
        for d in dirs:
            symbols = d.strip().split()
            dcode = symbols[0]
            if dcode == ".reg":
                decl = re.search(r'%([^<]*)<([^>]+)>', symbols[2])
                if not decl: continue
                prefix, virtual = decl.group(1), decl.group(2)
                for i in range(int(virtual)): registers[f"{prefix}{i}"]=Register(symbols[1])

        # 3. Parse instructions
        instructions = []
        def parse_operand(s):
            if s.startswith('['): return Addr(s[1:-1])
            elif s.startswith('%'): return Reg(s[1:])
            elif s.startswith('0f'): return Imm(struct.unpack('f', bytes.fromhex(s[2:]))[0])
            elif s.startswith('$'): return Label(s[1:])
            return Imm(int(s))

        pattern = re.compile(r'(?:(@\S+)\s+)?(\S+)\s*(.*)')
        for i in instrs:
            m = pattern.match(i)
            if not m: continue
            pred, op_str, operand_str = m.group(1), m.group(2), m.group(3)
            #print(pred, op_str, operand_str)
            parts = op_str.split('.')
            opcode, quals = parts[0], parts[1:]
            operands = [parse_operand(o.strip()) for o in operand_str.split(',') if o.strip()]
            instructions.append(Instruction(opcode, quals, operands, pred))
        return Kernel(params, instructions, registers)

def load_kernels(src) -> dict[str, Kernel]:
    src = re.sub(r'//.*', '', src)
    kstubs = []
    ENTRY = re.compile(r"""
        \.entry \s+ (?P<name>\w+)
        \s* \( (?P<params>[^)]*) \)
        \s* \{ (?P<body>[^}]*) \}
    """, re.VERBOSE | re.DOTALL)

    for m in ENTRY.finditer(src):
        param_str = m.group("params")
        kstubs.append((m.group("name"), param_str, m.group("body")))

    kernels = {}
    for (name, params, body) in kstubs:
        kernel = Kernel.decode(params, body)
        print(f"Kernel: {name}\n\t# params={len(kernel.params)}\n\t# registers={len(kernel.regfile)}\n\t# instructions={len(kernel.ops)}")
        kernels[name]=kernel
    return kernels

kernels = load_kernels(open("test/gemm.ptx", "r").read())
