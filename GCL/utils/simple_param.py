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
        with open(path, 'r') as f:
            content = f.read()
        return yaml.load(content, Loader=yaml.Loader)

    @staticmethod
    def _parse_json(path: str):
        with open(path, 'r') as f:
            content = f.read()
        return json.loads(content)

    @staticmethod
    def _mixin(orig_dict: dict, new_dict: dict):
        for k, new_v in new_dict.items():
            if k not in orig_dict:
                orig_dict[k] = new_v
            else:
                orig_v = orig_dict[k]
                if type(orig_v) is dict and type(new_v) is dict:
                    orig_v.update(new_v)
                else:
                    orig_dict[k] = new_v

    @staticmethod
    def _process_namespace(pre_loaded: dict, allowed_namespace: Optional[list] = None):
        namespace = []
        loaded = {}
        # Parse all keys beginning with`NAMESPACE:` to nested dicts
        for k, v in pre_loaded.items():
            ks = k.split(':')
            if len(ks) > 1:
                classname = ks[0]
                subkey = ks[1]
                namespace.append(classname)
                if classname in loaded and type(loaded[classname]) is dict:
                    loaded[classname][subkey] = v
                else:
                    loaded[classname] = {subkey: v}
            else:
                if type(v) is dict:
                    namespace.append(k)
                SimpleParam._mixin(loaded, {k: v})

        if allowed_namespace is not None:
            # Filter out not wanted namespaces
            loaded = {k: v for k, v in loaded.items() if type(v) is not dict or k in allowed_namespace}
            namespace = list(set(namespace).intersection(allowed_namespace))
        return namespace, loaded

    def __init__(self, default: Optional[dict] = None, allowed_namespaces: Optional[list] = None):
        default = default if default is not None else dict()
        namespace, loaded = self._process_namespace(default, allowed_namespaces)
        self.namespace = namespace
        self.parameters = loaded

    def __call__(self, allowed_namespaces: Optional[list] = None):
        if allowed_namespaces is None:
            return self.parameters

        loaded = {k: v for k, v in self.parameters.items() if k in allowed_namespaces}
        return loaded

    def load(self, preloaded: dict, allowed_namespaces: Optional[List] = None, *args, **kwargs):
        namespace, loaded = self._process_namespace(preloaded, allowed_namespaces)
        self.namespace = list(set(self.namespace).union(namespace))
        self._mixin(self.parameters, loaded)

    def update(self, from_: str, allowed_namespaces: Optional[list] = None, *args, **kwargs):
        if from_ == 'nni':
            loaded = nni.get_next_parameter()
            self.load(loaded, allowed_namespaces)
        else:
            loaded = dict()
            if from_.endswith('.json'):
                loaded = self._parse_json(from_)
            elif from_.endswith('.yaml') or from_.endswith('.yml'):
                loaded = self._parse_yaml(from_)

            if 'preprocess_nni' in kwargs and kwargs['preprocess_nni']:
                loaded = self._preprocess_nni(loaded)

            self.load(loaded, allowed_namespaces)


if __name__ == '__main__':
    sp = SimpleParam()
    print(sp.parameters)

    sp.load({'foo': {'bar': 1}})
    print(sp.parameters)

    sp.load({'foo:baz': 2})
    print(sp.parameters)

    sp.load({'foo:bar': '1', 'foo:baz': 3, 'foo': {'quux': 4}})
    print(sp.parameters)
