from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_DEST = Path("third_party") / "optitrack_natnet_sdk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an installed official OptiTrack NatNet SDK sample folder into "
            "third_party/ for local use. The destination is ignored by git."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Path to the downloaded/extracted NatNet SDK root, Samples folder, "
            "Python sample folder, or NatNetClient.py."
        ),
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    return parser.parse_args()


def find_sample_root(source: Path) -> Path:
    source = source.resolve()
    if source.is_file():
        return source.parent
    if (source / "NatNetClient.py").exists() or (source / "PythonSample.py").exists():
        return source

    for candidate in source.rglob("NatNetClient.py"):
        return candidate.parent
    for candidate in source.rglob("PythonSample.py"):
        return candidate.parent

    raise FileNotFoundError(
        "Could not find NatNetClient.py or PythonSample.py under "
        f"{source}. Download/extract the official NatNet SDK first."
    )


def copy_sample(source: Path, dest: Path) -> Path:
    sample_root = find_sample_root(source)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "Samples" / "Python"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(sample_root, target)

    return target


def main() -> None:
    args = parse_args()
    target = copy_sample(args.source, args.dest)
    print(f"Copied official NatNet Python sample to: {target}")
    if (target / "NatNetClient.py").exists():
        print("Recorder can use it automatically or with:")
        print(f'  --optitrack_natnet_path "{target}"')
    elif (target / "PythonSample.py").exists():
        print(
            "This SDK provides PythonSample.py direct depacketization. "
            "The recorder's default raw_udp mode already follows this approach."
        )


if __name__ == "__main__":
    main()
