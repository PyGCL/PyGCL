from enum import Enum
from typing import Optional, List
from HC.typechecking.types import Type, StructuralType, PrimitiveType
from HC.typechecking.typecheck_error import TypeCheckError, TypeMismatch, InvalidField, InvalidEnumValue


def check_type(x, tp: Type) -> Optional[TypeCheckError]:
    def construct_dict(path: List[str], v) -> dict:
        if len(path) == 1:
            return {path[0]: v}
        return construct_dict(path[:-1], {path[-1]: v})

    def recur(x, tp: Type, path: List[str]) -> Optional[TypeCheckError]:
        def check_struct(tp: StructuralType) -> Optional[TypeCheckError]:
            if not isinstance(x, dict):
                return TypeMismatch(path=path, expect=tp, actual=type(x))

            # x is a instance of dict
            dict_x: dict = x
            for k, v in dict_x.items():
                if len(k.split(':')) > 1:
                    # handle path-like key
                    ks = k.split(':')
                    d = construct_dict(ks, v)
                    err = recur(d, tp, path=path)
                else:
                    # normal key
                    if k not in tp.fields.keys():
                        return InvalidField(path=path, field_name=k, struct=tp)
                    err = recur(v, tp.fields[k], path=path + [k])
                if err is not None:
                    return err
            return None

        def check_primitive(tp: PrimitiveType) -> Optional[TypeCheckError]:
            if isinstance(x, tp.tp):
                return None
            elif issubclass(tp.tp, Enum):
                try:
                    x1 = tp.tp(x)
                    return None
                except ValueError as e:
                    return InvalidEnumValue(path=path, msg=f'{e}')
            else:
                return TypeMismatch(path=path, expect=tp, actual=type(x))

        return tp.pmatch(check_struct, check_primitive)

    return recur(x, tp, path=[])


if __name__ == '__main__':
    from dataclasses import dataclass
    from enum import Enum
    from HC.typechecking.utils import from_python_type, extract_valid_paths
    from HC.typechecking.types import show_type

    class Loss(Enum):
        InfoNCE = 'infonce'
        Debiased = 'debiased_infonce'

    @dataclass
    class DebiasedInfoNCE:
        tau: float
        tau_plus: float = 10.0

    @dataclass
    class Config:
        loss: Loss
        debiased_infonce: DebiasedInfoNCE

    config_tp = from_python_type(Config)
    print(extract_valid_paths(config_tp))
