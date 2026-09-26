#!/usr/bin/env python3
"""Entry point: `python src/run_pipeline.py run --data-dir <path to student_resource>`.

Adds this folder to sys.path so the `ber` package imports without installation or PYTHONPATH.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
