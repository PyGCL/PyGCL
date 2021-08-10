import argparse

from .search_space import gen_search_space


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    space_p = subparsers.add_parser('genspace')
    space_p.add_argument('--model', '-m', type=str, required=True,
                        help='Path of the configuration model class. Example: `train_config.ExpConfig`.')
    space_p.add_argument('--output', '-o', type=str, default='search_space.json',
                        help='Output path for the NNI search space file.')

    args = parser.parse_args()

    if args.command == 'genspace':
        gen_search_space(model=args.model, output_path=args.output)
