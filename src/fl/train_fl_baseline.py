"""
Dispatcher script to choose Phase 1 or Phase 2 training.

Usage:
    python -m src.fl.train_fl_baseline           # interactive choice
    python -m src.fl.train_fl_baseline 1         # force Phase 1
    python -m src.fl.train_fl_baseline 2         # force Phase 2
"""

import sys

from .train_phase1_small import main as main_phase1
from .train_phase2_agnews_sparse import main as main_phase2


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in {"1", "2"}:
        choice = sys.argv[1]
    else:
        print("Select training phase to run:")
        print("  1) Phase 1 - Small corpus, single global model (fast)")
        print("  2) Phase 2 - AG News, dense vs sparse + comm stats (slower)")
        choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        main_phase1()
    elif choice == "2":
        main_phase2()
    else:
        print("Invalid choice. Please run again with '1' or '2'.")


if __name__ == "__main__":
    main()


