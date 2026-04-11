import argparse
import json
import os
from typing import Dict, Set, TextIO, Tuple


def _read_list(path: str) -> Set[str]:
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _open_out(path: str, overwrite: bool) -> TextIO:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w")


def make_splits(apollo_root: str, overwrite: bool = False) -> None:
    label_path = os.path.join(apollo_root, "laneline_label.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing {label_path}")

    splits = ["standard", "illus_chg", "rare_subset"]

    # Prepare selection sets and output handles.
    targets: Dict[Tuple[str, str], Set[str]] = {}
    handles: Dict[Tuple[str, str], TextIO] = {}
    expected: Dict[Tuple[str, str], int] = {}
    written: Dict[Tuple[str, str], int] = {}

    for split in splits:
        for phase in ("train", "test"):
            list_path = os.path.join(apollo_root, "data_lists", split, f"{phase}.txt")
            if not os.path.exists(list_path):
                raise FileNotFoundError(f"Missing {list_path}")
            s = _read_list(list_path)
            targets[(split, phase)] = s
            expected[(split, phase)] = len(s)
            written[(split, phase)] = 0

            out_path = os.path.join(apollo_root, "data_splits", split, f"{phase}.json")
            handles[(split, phase)] = _open_out(out_path, overwrite=overwrite)

    # One-pass streaming over the big jsonlines file.
    with open(label_path, "r") as src:
        for idx, line in enumerate(src, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON on line {idx} of {label_path}") from e

            raw_file = obj.get("raw_file")
            if not raw_file:
                continue

            for split in splits:
                if raw_file in targets[(split, "train")]:
                    handles[(split, "train")].write(line + "\n")
                    written[(split, "train")] += 1
                if raw_file in targets[(split, "test")]:
                    handles[(split, "test")].write(line + "\n")
                    written[(split, "test")] += 1

            if idx % 20000 == 0:
                print(f"processed {idx} label lines")

    for h in handles.values():
        h.close()

    print("Done. Written counts:")
    for split in splits:
        for phase in ("train", "test"):
            e = expected[(split, phase)]
            w = written[(split, phase)]
            print(f"  {split}/{phase}: {w} (expected list size {e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ApolloSim data_splits from laneline_label.json + data_lists")
    parser.add_argument("apollo_root", help="ApolloSim root, e.g. data/Apollosim")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data_splits/*.json")
    args = parser.parse_args()
    make_splits(args.apollo_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
