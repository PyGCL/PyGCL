from __future__ import annotations
from typing import Dict, List
from dataclasses import field, dataclass, is_dataclass, fields
import yaml
import json


@dataclass
class ParameterSpace:
    space_type: str
    space_value: list

    def as_dict_nni(self):
        return {
            '_type': self.space_type,
            '_value': self.space_value
        }


@dataclass
class SearchSpace:
    parameters: Dict[str, ParameterSpace]

    def as_dict_nni(self):
        return {k: v.as_dict_nni() for k, v in self.parameters.items()}

    def as_json_nni(self):
        data = self.as_dict_nni()
        s = json.dumps(data, indent=4)
        return s

    def as_yaml_nni(self):
        data = self.as_dict_nni()
        s = yaml.dump(data)
        return s

    def __add__(self, other: SearchSpace):
        merged_parameters = {**self.parameters, **other.parameters}
        return SearchSpace(parameters=merged_parameters)


def with_search_space(default, space_type: str, space_value: list):
    return field(default=default,
                 metadata={'parameter_space': ParameterSpace(space_type=space_type, space_value=space_value)})


def extract_search_space(model: type) -> SearchSpace:
    def recur(t: type, path: List[str]):
        assert is_dataclass(t), 'extract_search_space should be called on dataclass.'
        res = SearchSpace(parameters=dict())
        fs = fields(t)
        for f in fs:
            name = f.name
            t = f.type
            meta = f.metadata
            if is_dataclass(t):
                res = res + recur(t, path + [name])
            elif 'parameter_space' in meta:
                res.parameters.update({f'{":".join(path)}:{name}': meta['parameter_space']})

        return res

    return recur(model, path=[])
