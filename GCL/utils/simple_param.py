from typing import List
import nni
import json
import yaml

from typing import Optional


class SimpleParam:
    @staticmethod
    def _preprocess_nni(params: dict):
        return {k.split('/')[1]: v for k, v in params.items()}

    @staticmethod
    def _parse_yaml(path: str):
        content = open(path).read()
        return yaml.load(content, Loader=yaml.Loader)

    @staticmethod
    def _parse_json(path: str):
        content = open(path).read()
        return json.loads(content)

    @staticmethod
    def merge_args(schema: List[str], args: object, param: dict) -> dict:
        args = { k: v for k, v in [(k, getattr(args, k)) for k in schema] if v is not None }
        return { **param, **args }

    def __init__(self, default: Optional[dict] = None):
        self.default = default if default is not None else dict()

    def __call__(self, from_: Optional[str] = 'None', *args, **kwargs):
        if from_ == 'nni':
            return {**self.default, **nni.get_next_parameter()}
        elif from_ != 'None':
            if from_.endswith('.json'):
                loaded = self._parse_json(from_)
            elif from_.endswith('.yaml') or from_.endswith('.yml'):
                loaded = self._parse_yaml(from_)
            else:
                raise NotImplementedError

            if 'preprocess_nni' in kwargs and kwargs['preprocess_nni']:
                loaded = self._preprocess_nni(loaded)

            return {**self.default, **loaded}

        return self.default
