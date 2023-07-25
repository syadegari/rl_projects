import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arg1",
        action="store_true",
        default=False,
        help="The help string for arg1. Type: %(type)s, Default: %(default)s",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
