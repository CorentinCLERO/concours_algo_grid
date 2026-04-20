from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solvers.greedy_montecarlo_joker import cli


def main() -> None:
    cli(default_dataset=1, description="Greedy + Monte Carlo optimizer for dataset_1")


if __name__ == "__main__":
    main()
