"""Read-only verification of archived bytes, optional originals and Git index."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--git-index', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    manifest = json.loads((root/'SNAPSHOT.json').read_text(encoding='utf-8'))
    total = 0
    for relative, expected in manifest['files'].items():
        parts = PurePosixPath(relative)
        if parts.is_absolute() or '..' in parts.parts or parts.parts[0] not in {'runtime','configs','releases'}:
            raise ValueError(f'Invalid payload path: {relative}')
        data = root.joinpath(*parts.parts).read_bytes()
        if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
            raise ValueError(f'Archive mismatch: {relative}')
        if args.source_root and args.source_root.joinpath(*parts.parts).read_bytes() != data:
            raise ValueError(f'Original source mismatch: {relative}')
        if args.git_index:
            tracked = root.joinpath(*parts.parts).relative_to(repo).as_posix()
            staged = subprocess.check_output(['git','show',':'+tracked], cwd=repo)
            if staged != data:
                raise ValueError(f'Git index changed payload bytes: {relative}')
        total += len(data)
    print(json.dumps(dict(passed=True, files=len(manifest['files']), bytes=total,
                          originals_checked=bool(args.source_root), git_index_checked=args.git_index)))


if __name__ == '__main__':
    main()
