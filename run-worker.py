import argparse

from solvers.greedy_montecarlo_joker import run_solver as run_solver_with_joker
from solvers.solver_dataset_4 import run_solver as run_solver_dataset_4
from solvers.solver_dataset_3 import run_solver as run_solver_dataset_3


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatcher vers le solver greedy + monte carlo du dataset choisi")
    parser.add_argument("--dataset", type=int, required=True, choices=[1, 2, 3, 4], help="Dataset number (1, 2, 3, or 4)")
    parser.add_argument("--iterations", type=int, default=5000, help="MC iterations per restart")
    parser.add_argument("--restarts", type=int, default=6, help="Number of random restarts")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de processus CPU (1 = desactive le parallelisme)")
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_<dataset>.txt")
    args = parser.parse_args()

    if args.dataset in (1, 2):
        run_solver_with_joker(
            dataset_id=args.dataset,
            iterations=args.iterations,
            restarts=args.restarts,
            workers=args.workers,
            write_solution=args.write,
        )
        return

    if args.dataset == 4:
        run_solver_dataset_4(
            dataset_id=args.dataset,
            iterations=args.iterations,
            restarts=args.restarts,
            workers=args.workers,
            write_solution=args.write,
        )
        return

    run_solver_dataset_3(
        dataset_id=args.dataset,
        iterations=args.iterations,
        restarts=args.restarts,
        workers=args.workers,
        write_solution=args.write,
    )


if __name__ == "__main__":
    main()

