from lark import Discard, Lark, Transformer, v_args
from typing import Optional
from dataclasses import dataclass

grammar = r"""
start: statement+

statement: directive
        |  kernel

directive: directive_name directive_value
directive_name: /\.\w+/
directive_value: /\S+/

kernel: ".visible" ".entry"  kernel_name params ")" "{" kernel_body* "}"
kernel_name: /\S+/
params: (param ("," param)*)?
param: ".param" /\.\w+/ /\w+/

kernel_body: "." variable ";"
        | "." kernel_directive ";"
        | ("@" predicate)? opcode ("." qualifier)* (operand ("," operand)*)? ";" -> instruction
        | "$" /\w+/ ":" -> label

variable: /[^.]+/ "." /[^%]+/ "%"  /[^<]+/ "<" /\d+/ ">"
kernel_directive: /[^;]+/

predicate: "%" "p" /\d+/
opcode: /\w+/
qualifier: /\w+/
operand: "%" /[\w.]+/ -> register_operand
    | "$" /\w+/ -> label_operand
    | "[" /[^\]]+/ "]" -> address_operand
    | immediate

immediate: /\-?\d+/ | /0f[0-9a-fA-F]+/

%ignore /\/\/.*/
%ignore /\/\*.*?\*\//s

%import common.WS
%ignore WS
"""

@dataclass
class Instruction:
    opcode: str
    qualifiers: list[str]
    operands: list
    predicate: Optional[str] = None

@dataclass
class Kernel:
    name: str
    params: list[str]
    regs: list[tuple]
    instructions: list[Instruction]

@v_args(inline=True)
class IRTransformer(Transformer):
    def register_operand(self, t):      return ("reg", str(t))
    def address_operand(self, t):       return ("addr", str(t).strip())
    def label_operand(self, t):         return ("label", str(t))
    def opcode(self, t):                return str(t)
    def qualifier(self, t):             return str(t)
    def predicate(self, t):             return f"%p{t}"
    def kernel_name(self, t):           return str(t).rstrip("(")
    def param(self, fmt, name):         return (str(fmt).lstrip("."), str(name))
    def variable(self, _, fmt, pfx, n): return (str(fmt).strip(), str(pfx), int(n))
    def kernel_directive(self, _):      return Discard
    def directive(self, *_):            return Discard
    def kernel_body(self, *args):
        if isinstance(args, tuple) and len(args) == 1: return args[0]
        return args
    def params(self, *args):            return [*args]
    def operand(self, opr):             return opr
    def label(self, name): return ("label", str(name))
    def immediate(self, t):
        if len(t) > 1 and t[1].lower() == 'f': return ("float", float(t[2:]))
        else: return ("int", int(t))

    @v_args(inline=False)
    def instruction(self, c):
        pred = c.pop(0) if c and isinstance(c[0], str) and c[0].startswith("%p") else None
        opcode = c.pop(0)
        quals = [x for x in c if isinstance(x, str)]
        operands = [x for x in c if not isinstance(x, str)]
        return Instruction(opcode, quals, operands, pred)

    @v_args(inline=False)
    def kernel(self, c):
        name, params, *raw = c
        regs = [b for b in raw if isinstance(b, tuple) and len(b) > 0 and b[0] != "label"]
        mixed = [b for b in raw if b not in regs]

        instructions, labels = [], {}
        for item in mixed:
            if isinstance(item, tuple) and len(item) == 0: continue
            if isinstance(item, tuple) and item[0] == "label": labels[item[1]]=len(instructions)
            elif isinstance(item, Instruction): instructions.append(item)
        # resolve label operands
        for op in instructions:
            for i, opr in enumerate(op.operands):
                if opr[0] == "label": op.operands[i] = ("pc", labels[opr[1]])
        return Kernel(name, params, regs, instructions)

    @v_args(inline=False)
    def statement(self, c): return c[0] if c else Discard
    @v_args(inline=False)
    def start(self, args):
        return { k.name : k for k in args if isinstance(k, Kernel) }

def ptx_to_ir(src: str):
    return IRTransformer().transform(Lark(grammar).parse(src))
