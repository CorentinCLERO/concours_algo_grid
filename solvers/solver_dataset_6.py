from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_solution

Grid = List[List[int]]
RectAction = Tuple[int, int, int, int, int]
JokerAction = Tuple[str, int, int, int, int, int]
Action = RectAction | JokerAction

RECT_STRATEGIES = ("area", "wide", "tall", "balanced")


def load_dataset(dataset_path: Path) -> dict:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def color_frequencies(grid: Grid) -> dict[int, int]:
    freq: dict[int, int] = {}
    for row in grid:
        for color in row:
            freq[color] = freq.get(color, 0) + 1
    return freq


def action_to_text(action: Action) -> str:
    if len(action) == 5:
        x1, y1, x2, y2, color = action
        return f"RECT {x1} {y1} {x2} {y2} {color}"
    _, x1, y1, x2, y2, _ = action
    return f"JOKER {x1} {y1} {x2} {y2}"


def actions_to_solution_text(actions: Sequence[Action]) -> str:
    return "\n".join(action_to_text(action) for action in actions)


def evaluate_actions(actions: Sequence[Action], dataset: dict) -> Tuple[int, bool, str]:
    solution_txt = actions_to_solution_text(actions)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    return test_solution.get_solution_score(solution_txt, dataset_txt)


def build_residual_rectangles(target: Grid, base_color: int) -> List[RectAction]:
    h = len(target)
    w = len(target[0])
    active: dict[Tuple[int, int, int], List[int]] = {}
    rects: List[RectAction] = []

    for y in range(h):
        row = target[y]
        runs: List[Tuple[int, int, int]] = []
        x = 0
        while x < w:
            if row[x] == base_color:
                x += 1
                continue

            color = row[x]
            x2 = x
            while x2 + 1 < w and row[x2 + 1] == color:
                x2 += 1
            runs.append((x, x2, color))
            x = x2 + 1

        next_active: dict[Tuple[int, int, int], List[int]] = {}
        for x1, x2, color in runs:
            key = (x1, x2, color)
            if key in active and active[key][3] == y - 1:
                run = active[key]
                run[3] = y
                next_active[key] = run
            else:
                next_active[key] = [x1, y, x2, y]

        for key, run in active.items():
            if key not in next_active:
                rects.append((run[0], run[1], run[2], run[3], key[2]))

        active = next_active

    for key, run in active.items():
        rects.append((run[0], run[1], run[2], run[3], key[2]))

    return rects


def rect_area(rect: RectAction) -> int:
    x1, y1, x2, y2, _ = rect
    return (x2 - x1 + 1) * (y2 - y1 + 1)


def rect_sort_key(rect: RectAction, strategy: str) -> Tuple[float, float, float, float, float]:
    x1, y1, x2, y2, color = rect
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    area = width * height

    if strategy == "wide":
        primary = float(width)
    elif strategy == "tall":
        primary = float(height)
    elif strategy == "balanced":
        primary = float(min(width, height))
    else:
        primary = float(area)

    return (primary, float(area), float(width), float(height), float(-color))


def mark_rectangles(h: int, w: int, rects: Sequence[RectAction]) -> List[List[bool]]:
    marked = [[False] * w for _ in range(h)]
    for x1, y1, x2, y2, _ in rects:
        for y in range(y1, y2 + 1):
            row = marked[y]
            for x in range(x1, x2 + 1):
                row[x] = True
    return marked


def build_residual_mask(target: Grid, base_color: int, already_corrected: List[List[bool]]) -> List[List[int]]:
    h = len(target)
    w = len(target[0])
    residual = [[0] * w for _ in range(h)]
    for y in range(h):
        row = target[y]
        corrected_row = already_corrected[y]
        residual_row = residual[y]
        for x in range(w):
            if not corrected_row[x] and row[x] != base_color:
                residual_row[x] = 1
    return residual


def build_prefix_sum(mask: List[List[int]]) -> List[List[int]]:
    h = len(mask)
    w = len(mask[0])
    prefix = [[0] * (w + 1) for _ in range(h + 1)]
    for y in range(h):
        run = 0
        prow = prefix[y + 1]
        prev = prefix[y]
        row = mask[y]
        for x in range(w):
            run += row[x]
            prow[x + 1] = prev[x + 1] + run
    return prefix


def area_sum(prefix: List[List[int]], x1: int, y1: int, x2: int, y2: int) -> int:
    return prefix[y2 + 1][x2 + 1] - prefix[y1][x2 + 1] - prefix[y2 + 1][x1] + prefix[y1][x1]


def joker_shapes(max_joker_size: int) -> List[Tuple[int, int]]:
    # Shapes tuned for area<=100 while capturing strips and square-ish blobs.
    base = [(10, 10), (5, 20), (20, 5), (4, 25), (25, 4), (2, 50), (50, 2), (1, 100), (100, 1)]
    return [(h, w) for h, w in base if h * w <= max_joker_size]


def select_grid_jokers_fixed10(
    target: Grid,
    base_color: int,
    already_corrected: List[List[bool]],
    joker_limit: int,
    max_joker_size: int,
) -> List[JokerAction]:
    if joker_limit <= 0:
        return []

    h = len(target)
    w = len(target[0])
    block = 10 if max_joker_size >= 100 else max(1, int(max_joker_size ** 0.5))
    candidates: List[Tuple[int, int, int, int, int]] = []

    for y1 in range(0, h, block):
        y2 = min(h - 1, y1 + block - 1)
        for x1 in range(0, w, block):
            x2 = min(w - 1, x1 + block - 1)
            gain = 0
            for y in range(y1, y2 + 1):
                row = target[y]
                corrected_row = already_corrected[y]
                for x in range(x1, x2 + 1):
                    if not corrected_row[x] and row[x] != base_color:
                        gain += 1
            if gain > 0:
                candidates.append((gain, x1, y1, x2, y2))

    candidates.sort(reverse=True)
    selected = candidates[:joker_limit]
    return [("JOKER", x1, y1, x2, y2, -1) for _gain, x1, y1, x2, y2 in selected]


def select_best_joker_blocks(
    target: Grid,
    base_color: int,
    already_corrected: List[List[bool]],
    joker_limit: int,
    max_joker_size: int,
) -> List[JokerAction]:
    if joker_limit <= 0:
        return []

    h = len(target)
    w = len(target[0])
    residual = build_residual_mask(target, base_color, already_corrected)
    prefix = build_prefix_sum(residual)

    cap = max(2000, joker_limit * 12)
    top_heap: List[Tuple[int, int, int, int, int]] = []
    for block_h, block_w in joker_shapes(max_joker_size):
        y_offsets = (0,) if block_h == 1 else (0, block_h // 2)
        x_offsets = (0,) if block_w == 1 else (0, block_w // 2)

        for oy in y_offsets:
            for ox in x_offsets:
                for y1 in range(oy, h - block_h + 1, block_h):
                    y2 = y1 + block_h - 1
                    for x1 in range(ox, w - block_w + 1, block_w):
                        x2 = x1 + block_w - 1
                        gain = area_sum(prefix, x1, y1, x2, y2)
                        if gain <= 0:
                            continue
                        item = (gain, x1, y1, x2, y2)
                        if len(top_heap) < cap:
                            heapq.heappush(top_heap, item)
                        elif gain > top_heap[0][0]:
                            heapq.heapreplace(top_heap, item)

    ranked = sorted(top_heap, reverse=True)
    occupied = [[False] * w for _ in range(h)]
    selected: List[JokerAction] = []

    for _gain, x1, y1, x2, y2 in ranked:
        if len(selected) >= joker_limit:
            break

        overlap = False
        for y in range(y1, y2 + 1):
            row = occupied[y]
            for x in range(x1, x2 + 1):
                if row[x]:
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            continue

        for y in range(y1, y2 + 1):
            row = occupied[y]
            for x in range(x1, x2 + 1):
                row[x] = True
        selected.append(("JOKER", x1, y1, x2, y2, -1))

    return selected


def build_candidate_solution(dataset: dict, restart: int) -> Tuple[List[Action], dict]:
    target: Grid = dataset["grid"]
    h = len(target)
    w = len(target[0])
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    freq = color_frequencies(target)
    base_color = max(freq.items(), key=lambda item: item[1])[0]
    strategy = RECT_STRATEGIES[restart % len(RECT_STRATEGIES)]

    rect_ratio = [0.85, 0.65, 0.45, 0.30][restart % 4]
    rect_budget = max(1, min(max_actions - 1, int((max_actions - 1) * rect_ratio)))

    residual_rects = build_residual_rectangles(target, base_color)
    residual_rects.sort(key=lambda rect: rect_sort_key(rect, strategy), reverse=True)

    selected_rects = residual_rects[:rect_budget]
    corrected_by_rects = mark_rectangles(h, w, selected_rects)

    joker_budget = min(max_jokers, max(0, max_actions - 1 - len(selected_rects)))
    mode_index = (restart // len(RECT_STRATEGIES)) % 2
    joker_mode = "grid10" if mode_index == 0 else "adaptive"
    if joker_mode == "adaptive":
        selected_jokers = select_best_joker_blocks(
            target,
            base_color,
            corrected_by_rects,
            joker_budget,
            max_joker_size,
        )
    else:
        selected_jokers = select_grid_jokers_fixed10(
            target,
            base_color,
            corrected_by_rects,
            joker_budget,
            max_joker_size,
        )

    actions: List[Action] = [(0, 0, w - 1, h - 1, base_color)]
    actions.extend(selected_rects)
    actions.extend(selected_jokers)

    # Backfill with more rectangles if jokers do not consume all remaining slots.
    remaining_slots = max_actions - len(actions)
    if remaining_slots > 0 and len(selected_rects) < len(residual_rects):
        actions.extend(residual_rects[len(selected_rects) : len(selected_rects) + remaining_slots])

    stats = {
        "strategy": strategy,
        "base_color": base_color,
        "rectangles": len(selected_rects),
        "jokers": len(selected_jokers),
        "residual_candidates": len(residual_rects),
        "joker_mode": joker_mode,
    }
    return actions, stats


def run_single_restart(dataset: dict, restart: int) -> Tuple[List[Action], dict]:
    return build_candidate_solution(dataset, restart)


def find_best_solution(dataset: dict, restarts: int, workers: int) -> Tuple[List[Action], dict]:
    best_actions: List[Action] = []
    best_stats: dict = {}
    best_score = -1

    if restarts <= 1 or workers <= 1:
        for restart in range(restarts):
            actions, stats = run_single_restart(dataset, restart)
            score, ok, _ = evaluate_actions(actions, dataset)
            if ok and (score > best_score or (score == best_score and (not best_actions or len(actions) < len(best_actions)))):
                best_actions = actions
                best_stats = stats
                best_score = score
        return best_actions, best_stats

    cpu_count = os.cpu_count() or 1
    worker_count = min(max(1, workers), cpu_count, restarts)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(run_single_restart, dataset, restart): restart for restart in range(restarts)}
        for future in as_completed(futures):
            actions, stats = future.result()
            score, ok, _ = evaluate_actions(actions, dataset)
            if ok and (score > best_score or (score == best_score and (not best_actions or len(actions) < len(best_actions)))):
                best_actions = actions
                best_stats = stats
                best_score = score

    return best_actions, best_stats


def run_solver(dataset_id: int, iterations: int, restarts: int, workers: int, write_solution: bool, show_progress: bool = True) -> List[Action]:
    _ = iterations
    _ = show_progress

    if dataset_id != 6:
        raise ValueError("solver_dataset_6 ne supporte que --dataset 6")
    if restarts < 1:
        raise ValueError("--restarts doit etre >= 1")
    if workers < 1:
        raise ValueError("--workers doit etre >= 1")

    dataset_path = ROOT_DIR / "datasets" / "dataset_6.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    dataset = load_dataset(dataset_path)

    started = time.time()
    best_actions, stats = find_best_solution(dataset, restarts=restarts, workers=workers)
    elapsed = time.time() - started

    solution_txt = actions_to_solution_text(best_actions)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    score, is_valid, message = test_solution.get_solution_score(solution_txt, dataset_txt)

    print("---------------------------------")
    print("Dataset: dataset_6.json")
    print(f"Actions: {len(best_actions)}")
    print(f"Valid  : {is_valid}")
    print(f"Score  : {score:_}")
    print(f"Message: {message}")
    print(f"Temps  : {elapsed:.2f}s")
    print(f"Strategie gagnante: {stats.get('strategy', 'n/a')}")
    print(f"Mode joker: {stats.get('joker_mode', 'n/a')}")
    print(f"Base color: {stats.get('base_color', 'n/a')}")
    print(f"Rectangles utilises: {stats.get('rectangles', 0)}")
    print(f"Jokers utilises: {stats.get('jokers', 0)}")
    print("Approche: fond dominant + meilleurs rectangles exacts + jokers de rattrapage sur zones residuelles")
    print("---------------------------------")

    if write_solution:
        out = ROOT_DIR / "solutions" / "solution_6.txt"
        out.write_text(solution_txt + "\n", encoding="utf-8")
        print(f"Solution ecrite dans: {out}")

    return best_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Solver optimise pour dataset_6")
    parser.add_argument("--dataset", type=int, default=6, choices=[6], help="Dataset number (must be 6)")
    parser.add_argument("--iterations", type=int, default=0, help="Parametre de compatibilite")
    parser.add_argument("--restarts", type=int, default=6, help="Nombre de variantes evaluees")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de processus CPU")
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_6.txt")
    args = parser.parse_args()

    run_solver(
        dataset_id=args.dataset,
        iterations=args.iterations,
        restarts=args.restarts,
        workers=args.workers,
        write_solution=args.write,
        show_progress=True,
    )


if __name__ == "__main__":
    main()





