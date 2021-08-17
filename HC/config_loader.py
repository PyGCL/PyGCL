from typing import TypeVar, List
from dataclasses import is_dataclass, fields
from argparse import ArgumentParser
import json
import yaml
import nni

from HC.typechecking import *


class ConfigLoader(object):
    T = TypeVar('T')

    def __init__(self, model: T, config: str, disable_argparse: bool = False):
        self.config_type = from_python_type(model)
        self.config_raw_type = model
        self.config = config
        self.arg_parser = self.construct_arg_parser() if not disable_argparse else None

    @staticmethod
    def _parse_yaml(path: str):
        with open(path, 'r') as f:
            content = f.read()
        return yaml.load(content, Loader=yaml.Loader)

    @staticmethod
    def _parse_json(path: str):
        with open(path, 'r') as f:
            content = f.read()
        return json.loads(content)

    @staticmethod
    def expand_paths(d: dict) -> dict:
        def construct_dict(path: List[str], v) -> dict:
            if len(path) == 1:
                return {path[0]: v}
            return construct_dict(path[:-1], {path[-1]: v})

        res = dict()

        def recur(x: dict, d: dict):
            for k, v in d.items():
                ks = k.split(':')
                if len(ks) > 1:
                    expanded_v = construct_dict(ks, v)
                    recur(x, expanded_v)
                else:
                    if isinstance(v, dict):
                        if k not in x:
                            x[k] = dict()
                        recur(x[k], v)
                    else:
                        x[k] = v

        recur(res, d)
        return res

    def construct_dataclass(self, data: dict) -> T:
        def recur(x, target: type):
            if is_dataclass(target):
                dict_x: dict = x  # x must be a dictionary, if type-checking is sound
                fs = fields(target)
                fs = {f.name: f.type for f in fs}
                dict_x = {k: recur(v, fs[k]) for k, v in dict_x.items()}
                return target(**dict_x)
            else:  # primitive types
                return target(x)

        return recur(data, self.config_raw_type)

    def get_config(self) -> T:
        if self.arg_parser is not None:
            args = self.arg_parser.parse_args()
            config_path = args.config
        else:
            config_path = self.config

        def load_dict() -> dict:
            if config_path == 'nni':
                loaded = nni.get_next_parameter()
            elif config_path.endswith('.json'):
                loaded = self._parse_json(config_path)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                loaded = self._parse_yaml(config_path)
            else:
                raise ValueError(f'Invalid config path: {config_path}')

            return loaded

        loaded = load_dict()
        if self.arg_parser is not None:
            args_dict = args.__dict__
            args_dict = {k: v for k, v in args_dict.items() if v is not None}
            loaded = {**loaded, **args_dict}

        if 'config' in loaded:
            loaded.pop('config')

        type_err = check_type(loaded, self.config_type)
        if type_err is not None:
            raise RuntimeError(f'Type error in config:\n{type_err.show()}')

        loaded = self.expand_paths(loaded)

        return self.construct_dataclass(loaded)

    def __call__(self) -> T:
        return self.get_config()

    def construct_arg_parser(self) -> ArgumentParser:
        paths = extract_valid_paths(self.config_type)
        parser = ArgumentParser()
        parser.add_argument('--config', type=str, default=self.config)

        for k, v in paths:
            parser.add_argument(f'--{k}', type=v, nargs='?')

        return parser
