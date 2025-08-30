import argparse
import glob
import json
import os
from typing import List, Dict


def filter_file(path: str, min_acc: float, max_acc: float, inplace: bool = True, backup: bool = True) -> int:
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    def ok(x):
        try:
            a = x.get('accuracy', None)
            if a is None:
                return False
            a = float(a)
            return (min_acc <= a <= max_acc)
        except Exception:
            return False
    filtered = [x for x in data if ok(x)]
    if inplace:
        if backup:
            bak = path + ".bak"
            try:
                if not os.path.exists(bak):
                    with open(bak, 'w') as bf:
                        json.dump(data, bf, indent=2)
            except Exception:
                pass
        with open(path, 'w') as f:
            json.dump(filtered, f, indent=2)
    return len(filtered)


def main():
    p = argparse.ArgumentParser(description="Filter generated_data JSON by accuracy range")
    p.add_argument('--glob', type=str, default='data/generated_data_*.json', help='Glob of files to filter')
    p.add_argument('--min', dest='min_acc', type=float, default=0.2)
    p.add_argument('--max', dest='max_acc', type=float, default=0.8)
    p.add_argument('--no_backup', action='store_true', help='Do not write .bak originals')
    args = p.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        print("No files matched", args.glob)
        return
    for fp in files:
        n = filter_file(fp, args.min_acc, args.max_acc, inplace=True, backup=(not args.no_backup))
        print(f"Filtered {fp} -> {n} examples")


if __name__ == '__main__':
    main()

