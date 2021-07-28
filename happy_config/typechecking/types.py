from __future__ import annotations
from abc import ABC
from typing import Dict, Callable, TypeVar
from dataclasses import dataclass

X = TypeVar('X')


class Type(ABC):
    def pmatch(self, struct: Callable[[StructuralType], X], primitive: Callable[[PrimitiveType], X]) -> X:
        pass


@dataclass
class StructuralType(Type):
    fields: Dict[str, Type]

    def pmatch(self, struct: Callable[[StructuralType], X], primitive: Callable[[PrimitiveType], X]) -> X:
        return struct(self)


@dataclass
class PrimitiveType(Type):
    tp: type

    def pmatch(self, struct: Callable[[StructuralType], X], primitive: Callable[[PrimitiveType], X]) -> X:
        return primitive(self)


def show_type(tp: Type) -> str:
    def show_struct(struct: StructuralType) -> str:
        fields = [f'{k}: {show_type(t)}' for k, t in struct.fields.items()]
        fields_str = ', '.join(fields)
        return '{' + fields_str + '}'

    def show_primitive(primitive: PrimitiveType) -> str:
        return primitive.tp.__name__

    return tp.pmatch(show_struct, show_primitive)
