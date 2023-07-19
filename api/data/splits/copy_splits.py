import shutil
from argparse import ArgumentParser
from pathlib import Path

FILE = Path(__file__)
REPO_ROOT = FILE.parents[3]


def main(dst):
    print(f'-> Copying splits to "{dst}"...')
    shutil.copytree(REPO_ROOT/'api/data/splits', dst, dirs_exist_ok=True)
    (dst/FILE.name).unlink()


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to copy splits for all datasets.')
    parser.add_argument('dst', default=REPO_ROOT/'data', type=Path, help='Path to copy datasets to. ')
    args = parser.parse_args()

    main(args.dst)
