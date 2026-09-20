"""Download the public pi0 initialization and tokenizer into a local asset directory."""

import argparse
import os
from pathlib import Path

from openpi.shared.download import maybe_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OPENPI_DATA_HOME"] = str(args.output_dir.resolve())
    params = maybe_download("gs://openpi-assets/checkpoints/pi0_base/params", gs={"token": "anon"})
    tokenizer = maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    print(f"pi0 initialization: {params}")
    print(f"Tokenizer: {tokenizer}")


if __name__ == "__main__":
    main()
