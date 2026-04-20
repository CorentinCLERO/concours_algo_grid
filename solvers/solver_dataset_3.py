import argparse
import os
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_solution

Rect = Tuple[int, int, int, int, int]  # x1, y1, x2, y2, color
Grid = List[List[int]]


def load_dataset(dataset_path: Path) -> dict:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def color_frequencies(grid: Grid) -> Dict[int, int]:
    freq: Dict[int, int] = {}
    for row in grid:
        for c in row:
            freq[c] = freq.get(c, 0) + 1
    return freq


def build_greedy_initial_solution(target: Grid, base_color: int, color_order: Sequence[int]) -> List[Rect]:
    h = len(target)
    w = len(target[0])

    runs_by_color: Dict[int, List[Tuple[int, int, int, int, int]]] = {}

    for y, row in enumerate(target):
        x = 0
        while x < w:
            c = row[x]
            x2 = x
            while x2 + 1 < w and row[x2 + 1] == c:
                x2 += 1
            if c != base_color:
                runs_by_color.setdefault(c, []).append((x, y, x2, y, c))
            x = x2 + 1

    actions: List[Rect] = [(0, 0, w - 1, h - 1, base_color)]

    for c in color_order:
        runs = runs_by_color.get(c, [])
        runs.sort(key=lambda t: (t[0], t[2], t[1]))

        current = None
        for x1, y1, x2, y2, _ in runs:
            if current and current[0] == x1 and current[2] == x2 and y1 == current[3] + 1:
                current = (current[0], current[1], current[2], y2, c)
            else:
                if current is not None:
                    actions.append(current)
                current = (x1, y1, x2, y2, c)

        if current is not None:
            actions.append(current)

    return actions


def actions_to_solution_text(actions: Sequence[Rect]) -> str:
    return "\n".join(
        f"RECT {x1} {y1} {x2} {y2} {color}" for x1, y1, x2, y2, color in actions
    )


def is_perfect_against_target(actions: Sequence[Rect], target: Grid) -> bool:
    h = len(target)
    w = len(target[0])
    grid = [[0 for _ in range(w)] for _ in range(h)]

    for x1, y1, x2, y2, color in actions:
        for y in range(y1, y2 + 1):
            row = grid[y]
            for x in range(x1, x2 + 1):
                row[x] = color

    for y in range(h):
        if grid[y] != target[y]:
            return False
    return True


def optimize_with_monte_carlo(
    initial_actions: List[Rect],
    target: Grid,
    rng: random.Random,
    iterations: int,
) -> List[Rect]:
    actions = initial_actions[:]
    best = actions[:]

    for _ in range(iterations):
        if len(actions) <= 2:
            break

        i = rng.randrange(1, len(actions))
        color = actions[i][4]

        same_color_indices = [idx for idx in range(1, len(actions)) if idx != i and actions[idx][4] == color]
        if not same_color_indices:
            continue

        j = same_color_indices[rng.randrange(len(same_color_indices))]
        if i > j:
            i, j = j, i

        r1 = actions[i]
        r2 = actions[j]
        merged = (
            min(r1[0], r2[0]),
            min(r1[1], r2[1]),
            max(r1[2], r2[2]),
            max(r1[3], r2[3]),
            color,
        )

        candidate = actions[:i] + [merged] + actions[i + 1 : j] + actions[j + 1 :]

        if is_perfect_against_target(candidate, target):
            actions = candidate
            if len(actions) < len(best):
                best = actions[:]

    return best


def pick_color_order(non_base_colors: List[int], freq: Dict[int, int], rng: random.Random, restart: int) -> List[int]:
    mode = restart % 4
    colors = non_base_colors[:]

    if mode == 0:
        return sorted(colors, key=lambda c: freq[c], reverse=True)
    if mode == 1:
        return sorted(colors, key=lambda c: freq[c])
    if mode == 2:
        rng.shuffle(colors)
        return colors

    preferred = [2, 0, 5, 7, 1]
    present = [c for c in preferred if c in colors]
    remaining = [c for c in colors if c not in present]
    rng.shuffle(remaining)
    return present + remaining


def run_single_restart(
    restart: int,
    target: Grid,
    base_color: int,
    non_base_colors: List[int],
    freq: Dict[int, int],
    iterations: int,
    seed: int,
) -> List[Rect]:
    rng = random.Random(seed + restart * 1009)
    color_order = pick_color_order(non_base_colors, freq, rng, restart)
    initial = build_greedy_initial_solution(target, base_color, color_order)
    return optimize_with_monte_carlo(
        initial_actions=initial,
        target=target,
        rng=rng,
        iterations=iterations,
    )


def find_best_solution(
    dataset: dict,
    iterations: int,
    restarts: int,
    seed: int,
    workers: int,
) -> List[Rect]:
    target: Grid = dataset["grid"]

    freq = color_frequencies(target)
    base_color = max(freq.keys(), key=lambda c: freq[c])
    non_base_colors = [c for c in freq if c != base_color]

    best_actions: List[Rect] = []
    best_count = 10**9

    if workers <= 1 or restarts <= 1:
        for restart in range(restarts):
            optimized = run_single_restart(
                restart=restart,
                target=target,
                base_color=base_color,
                non_base_colors=non_base_colors,
                freq=freq,
                iterations=iterations,
                seed=seed,
            )

            if len(optimized) < best_count:
                best_actions = optimized
                best_count = len(optimized)
        return best_actions

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_single_restart,
                restart,
                target,
                base_color,
                non_base_colors,
                freq,
                iterations,
                seed,
            )
            for restart in range(restarts)
        ]

        for future in futures:
            optimized = future.result()
            if len(optimized) < best_count:
                best_actions = optimized
                best_count = len(optimized)

    return best_actions


def run_solver(
    dataset_id: int,
    iterations: int,
    restarts: int,
    workers: int,
    write_solution: bool,
) -> List[Rect]:
    if workers < 1:
        raise ValueError("--workers doit etre >= 1")
    if restarts < 1:
        raise ValueError("--restarts doit etre >= 1")
    if iterations < 0:
        raise ValueError("--iterations doit etre >= 0")

    cpu_count = os.cpu_count() or 1
    worker_count = min(workers, cpu_count, restarts)
    run_seed = random.SystemRandom().randint(1, 2_147_483_647)

    project_root = ROOT_DIR
    dataset_path = project_root / "datasets" / f"dataset_{dataset_id}.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    dataset = load_dataset(dataset_path)

    started = time.time()
    best_actions = find_best_solution(
        dataset=dataset,
        iterations=iterations,
        restarts=restarts,
        seed=run_seed,
        workers=worker_count,
    )
    elapsed = time.time() - started

    solution_txt = actions_to_solution_text(best_actions)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    score, is_valid, message = test_solution.get_solution_score(solution_txt, dataset_txt)

    print("---------------------------------")
    print(f"Dataset: dataset_{dataset_id}.json")
    print(f"Actions: {len(best_actions)}")
    print(f"Valid  : {is_valid}")
    print(f"Score  : {score:_}")
    print(f"Message: {message}")
    print(f"Temps  : {elapsed:.2f}s")
    print(f"Seed run: {run_seed}")
    print(f"CPU workers utilises: {worker_count}")
    print("---------------------------------")

    if write_solution:
        out = project_root / "solutions" / f"solution_{dataset_id}.txt"
        out.write_text(solution_txt + "\n", encoding="utf-8")
        print(f"Solution ecrite dans: {out}")

    return best_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy + Monte Carlo optimizer for dataset_3")
    parser.add_argument("--dataset", type=int, default=3, help="Dataset number (default: 3)")
    parser.add_argument("--iterations", type=int, default=5000, help="MC iterations per restart")
    parser.add_argument("--restarts", type=int, default=5, help="Number of random restarts")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de processus CPU (1 = desactive le parallelisme)")
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_<dataset>.txt")
    args = parser.parse_args()

    run_solver(
        dataset_id=args.dataset,
        iterations=args.iterations,
        restarts=args.restarts,
        workers=args.workers,
        write_solution=args.write,
    )


if __name__ == "__main__":
    main()
