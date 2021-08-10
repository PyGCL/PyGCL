from __future__ import annotations
from typing import List, TypeVar, Callable
from abc import ABC
from dataclasses import dataclass
from happy_config.typechecking.types import Type, show_type, StructuralType

X = TypeVar('X')


class TypeCheckError(ABC):
    def pmatch(self, mismatch: Callable[[TypeMismatch], X], invalid_field: Callable[[InvalidField], X],
               invalid_enum: Callable[[InvalidEnumValue], X]) -> X:
        pass


@dataclass
class TypeMismatch(TypeCheckError):
    path: List[str]
    expect: Type
    actual: type

    def show(self) -> str:
        path_str = ':'.join(self.path)
        return f'At position [{path_str}]:\n  expect: {show_type(self.expect)}\n  found:  {self.actual}'

    def pmatch(self, mismatch: Callable[[TypeMismatch], X], invalid_field: Callable[[InvalidField], X],
               invalid_enum: Callable[[InvalidEnumValue], X]) -> X:
            return mismatch(self)


@dataclass
class InvalidField(TypeCheckError):
    path: List[str]
    field_name: str
    struct: StructuralType

    def show(self) -> str:
        path_str = ':'.join(self.path)
        return f'For {path_str}: invalid field {self.field_name} in structural type {show_type(self.struct)}'

    def pmatch(self, mismatch: Callable[[TypeMismatch], X], invalid_field: Callable[[InvalidField], X],
               invalid_enum: Callable[[InvalidEnumValue], X]) -> X:
        return invalid_field(self)


@dataclass
class InvalidEnumValue(TypeCheckError):
    path: List[str]
    msg: str

    def show(self) -> str:
        path_str = ':'.join(self.path)
        return f'For {path_str}: invalid enumerate value ({self.msg})'

    def pmatch(self, mismatch: Callable[[TypeMismatch], X], invalid_field: Callable[[InvalidField], X],
               invalid_enum: Callable[[InvalidEnumValue], X]) -> X:
        return invalid_enum(self)
