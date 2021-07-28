from pydoc import locate

from happy_config.param_tuning import extract_search_space


def gen_search_space(model: type, output_path: str, **kwargs):
    model_class = locate(model)
    search_space = extract_search_space(model_class)

    with open(output_path, 'w') as f:
        f.write(search_space.as_json_nni())
