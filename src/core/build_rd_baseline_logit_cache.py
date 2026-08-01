from __future__ import annotations

import argparse
import json
from pathlib import Path


LEGACY_CACHE_STATUS = "retired_incompatible_with_physical_rd_v2"


def build_baseline_logit_cache(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError(
        "legacy RD baseline logit cache is retired: it does not reproduce the active Physical RD input/time contract; "
        "do not rebuild or use it as an anchor residual cache"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retired legacy RD baseline logit cache command.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    try:
        build_baseline_logit_cache(checkpoint=Path(args.checkpoint) if args.checkpoint else None)
    except RuntimeError as error:
        print(json.dumps({"status": LEGACY_CACHE_STATUS, "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
