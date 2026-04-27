from lark import Lark, Transformer

grammar = r"""
start: statement+

statement: directive
        |  kernel

directive: directive_name directive_value
directive_name: /\.\w+/
directive_value: /\S+/

kernel: ".visible" ".entry"  /\S+\(/ params ")" "{" kernel_body* "}"
params: (param ("," param)*)?
param: ".param" /\.\w+/ /\w+/

kernel_body: "." /[^;]+/ ";" -> kernel_directive
        | op (operand ("," operand)*)? ";" -> instruction
        | "$" /\w+/ ":" -> label

op: ("@" predicate)? opcode ("." qualifier)*
predicate: "%" "p" /\d+/
opcode: /\w+/
qualifier: /\w+/
operand: "%" /[\w.]+/ -> register_operand
    | "$" /\w+/ -> label_operand
    | immediate
    | "[" /[^\]]+/ "]" -> address_operand

immediate: ("-")? /\d+/
    | "0f" /\w+/

%ignore /\/\/.*/
%ignore /\/\*.*?\*\//s

%import common.WS
%ignore WS
"""

class IRTransformer(Transformer):
    pass

input = open("test/gemm.ptx").read()

parser = Lark(grammar)
ast = parser.parse(input)
ir = IRTransformer().transform(ast)
print(ast)
