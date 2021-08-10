from typing import List, Tuple
from HC.typechecking.types import *
from dataclasses import is_dataclass, fields


def from_python_type(tp: type) -> Type:
    def from_dataclass(tp: type) -> Type:
        fs = fields(tp)
        fs = {f.name: from_python_type(f.type) for f in fs}
        return StructuralType(fields=fs)

    def from_primitive(tp: type) -> Type:
        return PrimitiveType(tp=tp)

    if is_dataclass(tp):
        return from_dataclass(tp)
    else:
        return from_primitive(tp)


def extract_valid_paths(tp: Type):
    def structural(struct: StructuralType):
        res = []
        for k, t in struct.fields.items():
            paths = extract_valid_paths(t)
            if isinstance(paths, type):
                res.append((k, paths))
            else:
                paths = [(f'{k}:{k0}', v0) for k0, v0 in paths]
                res = res + paths
        return res

    def primitive(primitive: PrimitiveType):
        return primitive.tp

    return tp.pmatch(structural, primitive)


if __name__ == '__main__':
    pass
