import argparse
import json
import os
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

Rect = Tuple[int, int, int, int, int]
Action = Tuple[str, int, int, int, int, int]
Grid = List[List[int]]


def load_dataset(dataset_path: Path) -> dict:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def color_frequencies(grid: Grid) -> Dict[int, int]:
    freq: Dict[int, int] = {}
    for row in grid:
        for c in row:
            freq[c] = freq.get(c, 0) + 1
    return freq


def action_to_text(action: Action) -> str:
    kind, x1, y1, x2, y2, color = action
    if kind == "RECT":
        return f"RECT {x1} {y1} {x2} {y2} {color}"
    return f"JOKER {x1} {y1} {x2} {y2}"


def actions_to_solution_text(actions: Sequence[Action]) -> str:
    return "\n".join(action_to_text(action) for action in actions)


def apply_action_in_place(grid: Grid, target: Grid, action: Action) -> None:
    kind, x1, y1, x2, y2, color = action
    if kind == "RECT":
        for y in range(y1, y2 + 1):
            row = grid[y]
            for x in range(x1, x2 + 1):
                row[x] = color
        return

    for y in range(y1, y2 + 1):
        row = grid[y]
        target_row = target[y]
        for x in range(x1, x2 + 1):
            row[x] = target_row[x]


def count_correct_cells(grid: Grid, target: Grid) -> int:
    total = 0
    for y, row in enumerate(grid):
        target_row = target[y]
        for x, val in enumerate(row):
            if val == target_row[x]:
                total += 1
    return total


def compute_gain_for_action(grid: Grid, target: Grid, action: Action) -> int:
    kind, x1, y1, x2, y2, color = action
    gain = 0

    for y in range(y1, y2 + 1):
        row = grid[y]
        target_row = target[y]
        for x in range(x1, x2 + 1):
            before_ok = row[x] == target_row[x]
            if kind == "RECT":
                after_val = color
            else:
                after_val = target_row[x]
            after_ok = after_val == target_row[x]
            gain += int(after_ok) - int(before_ok)

    return gain


def is_perfect_against_target(actions: Sequence[Action], target: Grid) -> bool:
    h = len(target)
    w = len(target[0])
    grid = [[0 for _ in range(w)] for _ in range(h)]

    for action in actions:
        apply_action_in_place(grid, target, action)

    return count_correct_cells(grid, target) == h * w


def count_jokers(actions: Sequence[Action]) -> int:
    return sum(1 for kind, *_ in actions if kind == "JOKER")


def best_row_run(target: Grid, x: int, y: int) -> Tuple[int, int]:
    color = target[y][x]
    w = len(target[0])

    left = x
    while left - 1 >= 0 and target[y][left - 1] == color:
        left -= 1

    right = x
    while right + 1 < w and target[y][right + 1] == color:
        right += 1

    return left, right


def expand_run_vertically(target: Grid, x1: int, x2: int, y: int, color: int) -> Tuple[int, int]:
    h = len(target)

    top = y
    while top - 1 >= 0 and all(target[top - 1][x] == color for x in range(x1, x2 + 1)):
        top -= 1

    bottom = y
    while bottom + 1 < h and all(target[bottom + 1][x] == color for x in range(x1, x2 + 1)):
        bottom += 1

    return top, bottom


def generate_rect_candidates(target: Grid, x: int, y: int, rng: random.Random) -> List[Rect]:
    color = target[y][x]
    left, right = best_row_run(target, x, y)
    top, bottom = expand_run_vertically(target, left, right, y, color)

    candidates: List[Rect] = [(left, top, right, bottom, color)]

    for _ in range(3):
        rx1 = rng.randint(left, x)
        rx2 = rng.randint(x, right)
        rtop, rbottom = expand_run_vertically(target, rx1, rx2, y, color)
        candidates.append((rx1, rtop, rx2, rbottom, color))

    dedup: Dict[Tuple[int, int, int, int, int], Rect] = {}
    for rect in candidates:
        dedup[rect] = rect
    return list(dedup.values())


def generate_joker_candidates(
    x: int,
    y: int,
    width: int,
    height: int,
    max_joker_size: int,
    rng: random.Random,
) -> List[Action]:
    if max_joker_size <= 0:
        return []

    candidates: List[Action] = []

    max_side_w = min(width, max_joker_size)
    max_side_h = min(height, max_joker_size)

    for _ in range(8):
        jw = rng.randint(1, max_side_w)
        jh = rng.randint(1, max_side_h)
        if jw * jh > max_joker_size:
            continue

        x1_min = max(0, x - jw + 1)
        x1_max = min(x, width - jw)
        y1_min = max(0, y - jh + 1)
        y1_max = min(y, height - jh)

        if x1_min > x1_max or y1_min > y1_max:
            continue

        x1 = rng.randint(x1_min, x1_max)
        y1 = rng.randint(y1_min, y1_max)
        x2 = x1 + jw - 1
        y2 = y1 + jh - 1

        candidates.append(("JOKER", x1, y1, x2, y2, -1))

    return candidates


def evaluate_actions(actions: Sequence[Action], dataset: dict) -> Tuple[int, bool, bool, int]:
    target: Grid = dataset["grid"]
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]

    if len(actions) == 0:
        return 0, False, False, 0

    if len(actions) > max_actions:
        return 0, False, False, 0

    if count_jokers(actions) > max_jokers:
        return 0, False, False, 0

    h = len(target)
    w = len(target[0])
    total_cells = h * w
    grid = [[0 for _ in range(w)] for _ in range(h)]

    for action in actions:
        apply_action_in_place(grid, target, action)

    correct = count_correct_cells(grid, target)
    perfect = correct == total_cells

    if perfect:
        score = round(max_actions / len(actions) * 1_000_000)
        return score, True, True, correct

    score = round(correct / total_cells * 1_000_000)
    return score, True, False, correct


def iter_all_joker_rectangles(width: int, height: int, max_area: int) -> List[Tuple[int, int, int, int]]:
    rects: List[Tuple[int, int, int, int]] = []
    for y1 in range(height):
        for y2 in range(y1, height):
            rect_h = y2 - y1 + 1
            for x1 in range(width):
                for x2 in range(x1, width):
                    rect_w = x2 - x1 + 1
                    if rect_w * rect_h <= max_area:
                        rects.append((x1, y1, x2, y2))
    return rects


def greedy_joker_prepass(
    actions: List[Action],
    grid: Grid,
    target: Grid,
    max_jokers: int,
    max_joker_size: int,
) -> int:
    if max_jokers <= 0 or max_joker_size <= 0:
        return 0

    h = len(target)
    w = len(target[0])
    _ = h
    _ = w
    used = 0
    all_jokers = iter_all_joker_rectangles(len(target[0]), len(target), max_joker_size)

    for _ in range(max_jokers):
        best_action: Action = ("RECT", 0, 0, 0, 0, 0)
        best_gain = 0
        best_area = 0

        for x1, y1, x2, y2 in all_jokers:
            candidate: Action = ("JOKER", x1, y1, x2, y2, -1)
            gain = compute_gain_for_action(grid, target, candidate)
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if gain > best_gain or (gain == best_gain and area > best_area):
                best_gain = gain
                best_area = area
                best_action = candidate

        if best_gain <= 1:
            break

        actions.append(best_action)
        apply_action_in_place(grid, target, best_action)
        used += 1

    return used


def generate_patch_candidates_for_grid(grid: Grid, target: Grid, max_joker_size: int, jokers_allowed: bool) -> List[Action]:
    h = len(target)
    w = len(target[0])
    candidates: List[Action] = []

    mismatch = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
    for x, y in mismatch:
        color = target[y][x]
        candidates.append(("RECT", x, y, x, y, color))

    if jokers_allowed and max_joker_size > 0:
        for x, y in mismatch:
            for x1 in range(max(0, x - 3), x + 1):
                for y1 in range(max(0, y - 3), y + 1):
                    for x2 in range(x, min(w - 1, x1 + 5) + 1):
                        for y2 in range(y, min(h - 1, y1 + 5) + 1):
                            area = (x2 - x1 + 1) * (y2 - y1 + 1)
                            if area <= max_joker_size:
                                candidates.append(("JOKER", x1, y1, x2, y2, -1))

    dedup: Dict[Action, Action] = {}
    for action in candidates:
        dedup[action] = action
    return list(dedup.values())


def repair_with_single_swap(actions: List[Action], dataset: dict) -> List[Action]:
    target: Grid = dataset["grid"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    score, is_valid, perfect, _ = evaluate_actions(actions, dataset)
    if not is_valid or perfect:
        return actions

    for idx in range(1, len(actions)):
        base = actions[:idx] + actions[idx + 1 :]
        if count_jokers(base) > max_jokers:
            continue

        h = len(target)
        w = len(target[0])
        grid = [[0 for _ in range(w)] for _ in range(h)]
        for action in base:
            apply_action_in_place(grid, target, action)

        jokers_allowed = count_jokers(base) < max_jokers
        for patch in generate_patch_candidates_for_grid(grid, target, max_joker_size, jokers_allowed):
            candidate = base + [patch]
            if len(candidate) != len(actions):
                continue
            if count_jokers(candidate) > max_jokers:
                continue
            if is_perfect_against_target(candidate, target):
                new_score, _, _, _ = evaluate_actions(candidate, dataset)
                if new_score >= score:
                    return candidate

    return actions


def greedy_construct(dataset: dict, rng: random.Random) -> List[Action]:
    target: Grid = dataset["grid"]
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    h = len(target)
    w = len(target[0])
    total_cells = h * w

    freq = color_frequencies(target)
    base_color = max(freq.keys(), key=lambda c: freq[c])

    actions: List[Action] = [("RECT", 0, 0, w - 1, h - 1, base_color)]
    grid = [[base_color for _ in range(w)] for _ in range(h)]

    jokers_used = greedy_joker_prepass(actions, grid, target, max_jokers, max_joker_size)
    correct = count_correct_cells(grid, target)

    while correct < total_cells and len(actions) < max_actions:
        mismatch_cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
        if not mismatch_cells:
            break

        sampled = mismatch_cells[:]
        rng.shuffle(sampled)
        sampled = sampled[: min(24, len(sampled))]

        best_action: Action = ("RECT", 0, 0, 0, 0, 0)
        best_value = -10**9
        best_gain = -10**9

        for x, y in sampled:
            for rx1, ry1, rx2, ry2, color in generate_rect_candidates(target, x, y, rng):
                candidate = ("RECT", rx1, ry1, rx2, ry2, color)
                gain = compute_gain_for_action(grid, target, candidate)
                value = float(gain)
                if value > best_value:
                    best_value = value
                    best_gain = gain
                    best_action = candidate

            if jokers_used < max_jokers:
                for candidate in generate_joker_candidates(x, y, w, h, max_joker_size, rng):
                    gain = compute_gain_for_action(grid, target, candidate)
                    area = (candidate[3] - candidate[1] + 1) * (candidate[4] - candidate[2] + 1)
                    value = gain + 0.08 * area
                    if value > best_value:
                        best_value = value
                        best_gain = gain
                        best_action = candidate

        if best_gain <= 0:
            break

        actions.append(best_action)
        apply_action_in_place(grid, target, best_action)
        correct += best_gain
        if best_action[0] == "JOKER":
            jokers_used += 1

    return actions


def action_inside(action: Action, x1: int, y1: int, x2: int, y2: int) -> bool:
    _, ax1, ay1, ax2, ay2, _ = action
    return x1 <= ax1 <= ax2 <= x2 and y1 <= ay1 <= ay2 <= y2


def try_joker_replacement(actions: List[Action], dataset: dict, rng: random.Random) -> List[Action]:
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]
    if max_jokers <= 0 or max_joker_size <= 0:
        return actions

    if count_jokers(actions) >= max_jokers:
        return actions

    if len(actions) < 3:
        return actions

    i = rng.randrange(1, len(actions))
    j = rng.randrange(1, len(actions))
    if i > j:
        i, j = j, i
    if i == j:
        return actions

    bbox_x1 = min(actions[k][1] for k in range(i, j + 1))
    bbox_y1 = min(actions[k][2] for k in range(i, j + 1))
    bbox_x2 = max(actions[k][3] for k in range(i, j + 1))
    bbox_y2 = max(actions[k][4] for k in range(i, j + 1))

    area = (bbox_x2 - bbox_x1 + 1) * (bbox_y2 - bbox_y1 + 1)
    if area > max_joker_size:
        return actions

    removable = [idx for idx, action in enumerate(actions) if action_inside(action, bbox_x1, bbox_y1, bbox_x2, bbox_y2)]
    if len(removable) <= 1:
        return actions

    removable_set = set(removable)
    candidate = [action for idx, action in enumerate(actions) if idx not in removable_set]
    candidate.append(("JOKER", bbox_x1, bbox_y1, bbox_x2, bbox_y2, -1))

    if len(candidate) >= len(actions):
        return actions

    if count_jokers(candidate) > max_jokers:
        return actions

    if is_perfect_against_target(candidate, dataset["grid"]):
        return candidate
    return actions


def optimize_with_monte_carlo(
    initial_actions: List[Action],
    dataset: dict,
    rng: random.Random,
    iterations: int,
) -> List[Action]:
    target: Grid = dataset["grid"]
    actions = initial_actions[:]
    best = actions[:]

    best_score, _, _, _ = evaluate_actions(best, dataset)

    for _ in range(iterations):
        if len(actions) <= 1:
            break

        op = rng.random()
        candidate = actions

        if op < 0.55 and len(actions) > 1:
            remove_idx = rng.randrange(1, len(actions))
            candidate = actions[:remove_idx] + actions[remove_idx + 1 :]
        elif op < 0.85:
            rect_indices = [idx for idx, action in enumerate(actions) if idx > 0 and action[0] == "RECT"]
            if len(rect_indices) >= 2:
                i = rect_indices[rng.randrange(len(rect_indices))]
                color = actions[i][5]
                same_color = [idx for idx in rect_indices if idx != i and actions[idx][5] == color]
                if same_color:
                    j = same_color[rng.randrange(len(same_color))]
                    if i > j:
                        i, j = j, i

                    r1 = actions[i]
                    r2 = actions[j]
                    merged: Action = (
                        "RECT",
                        min(r1[1], r2[1]),
                        min(r1[2], r2[2]),
                        max(r1[3], r2[3]),
                        max(r1[4], r2[4]),
                        color,
                    )
                    candidate = actions[:i] + [merged] + actions[i + 1 : j] + actions[j + 1 :]
        else:
            candidate = try_joker_replacement(actions, dataset, rng)

        if candidate is actions:
            continue

        score, is_valid, perfect, _ = evaluate_actions(candidate, dataset)
        if not is_valid:
            continue

        if perfect and is_perfect_against_target(candidate, target):
            actions = candidate
            if score > best_score or (score == best_score and len(candidate) < len(best)):
                best = candidate[:]
                best_score = score

    return best


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def tiny_beam_construct(dataset: dict, seed: int) -> List[Action]:
    target: Grid = dataset["grid"]
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    h = len(target)
    w = len(target[0])
    total_cells = h * w

    freq = color_frequencies(target)
    base_color = max(freq.keys(), key=lambda c: freq[c])
    base_action: Action = ("RECT", 0, 0, w - 1, h - 1, base_color)
    base_grid = [[base_color for _ in range(w)] for _ in range(h)]
    base_correct = count_correct_cells(base_grid, target)

    rng = random.Random(seed)
    beam_width = 120
    beam: List[Tuple[List[Action], Grid, int, int]] = [([base_action], base_grid, 0, base_correct)]
    best_actions: List[Action] = [base_action]
    best_correct = base_correct

    colors = sorted(freq.keys())
    all_rect_actions: List[Action] = []
    for y1 in range(h):
        for y2 in range(y1, h):
            for x1 in range(w):
                for x2 in range(x1, w):
                    for color in colors:
                        all_rect_actions.append(("RECT", x1, y1, x2, y2, color))

    all_joker_actions = [("JOKER", x1, y1, x2, y2, -1) for x1, y1, x2, y2 in iter_all_joker_rectangles(w, h, max_joker_size)]

    for _depth in range(1, max_actions):
        next_beam: List[Tuple[List[Action], Grid, int, int]] = []

        for actions, grid, jokers_used, correct in beam:
            if correct == total_cells:
                return actions

            mismatch = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
            if not mismatch:
                return actions

            rng.shuffle(mismatch)
            seeds = mismatch[: min(8, len(mismatch))]

            candidates: List[Action] = []
            for x, y in seeds:
                color = target[y][x]
                candidates.append(("RECT", x, y, x, y, color))

                for rx1, ry1, rx2, ry2, rcolor in generate_rect_candidates(target, x, y, rng):
                    candidates.append(("RECT", rx1, ry1, rx2, ry2, rcolor))

                if jokers_used < max_jokers:
                    candidates.extend(generate_joker_candidates(x, y, w, h, max_joker_size, rng))

            scored_rects: List[Tuple[int, Action]] = []
            for candidate in all_rect_actions:
                gain = compute_gain_for_action(grid, target, candidate)
                if gain > 0:
                    scored_rects.append((gain, candidate))
            scored_rects.sort(key=lambda t: t[0], reverse=True)

            candidates = [action for _gain, action in scored_rects[:80]]

            if len(scored_rects) > 80:
                tail = [action for _gain, action in scored_rects[80:]]
                rng.shuffle(tail)
                candidates.extend(tail[:20])

            if jokers_used < max_jokers:
                scored_jokers: List[Tuple[int, Action]] = []
                for candidate in all_joker_actions:
                    gain = compute_gain_for_action(grid, target, candidate)
                    if gain > 0:
                        scored_jokers.append((gain, candidate))
                scored_jokers.sort(key=lambda t: t[0], reverse=True)
                candidates.extend(action for _gain, action in scored_jokers[:40])

            if not candidates:
                continue

            dedup: Dict[Action, Action] = {action: action for action in candidates}

            for candidate in dedup.values():
                if candidate[0] == "JOKER" and jokers_used >= max_jokers:
                    continue

                next_grid = clone_grid(grid)
                gain = compute_gain_for_action(next_grid, target, candidate)
                apply_action_in_place(next_grid, target, candidate)
                next_correct = correct + gain

                if gain <= 0 and len(actions) + 2 < max_actions:
                    continue

                next_actions = actions + [candidate]
                next_jokers = jokers_used + int(candidate[0] == "JOKER")
                next_beam.append((next_actions, next_grid, next_jokers, next_correct))

                if next_correct > best_correct:
                    best_correct = next_correct
                    best_actions = next_actions
                    if best_correct == total_cells:
                        return best_actions

        if not next_beam:
            break

        next_beam.sort(key=lambda item: (item[3], -count_jokers(item[0]), -len(item[0])), reverse=True)

        pruned: List[Tuple[List[Action], Grid, int, int]] = []
        seen = set()
        for item in next_beam:
            key = tuple(tuple(row) for row in item[1])
            if key in seen:
                continue
            seen.add(key)
            pruned.append(item)
            if len(pruned) >= beam_width:
                break

        beam = pruned

    return best_actions


def run_single_restart(dataset: dict, restart: int, iterations: int, seed: int) -> List[Action]:
    rng = random.Random(seed + restart * 1013)
    target: Grid = dataset["grid"]
    tiny_problem = len(target) * len(target[0]) <= 64 and dataset["maxActions"] <= 12

    if tiny_problem:
        initial = tiny_beam_construct(dataset, seed + restart * 17)
    else:
        initial = greedy_construct(dataset, rng)

    optimized = optimize_with_monte_carlo(initial, dataset, rng, iterations)
    return repair_with_single_swap(optimized, dataset)


def find_best_solution(dataset: dict, iterations: int, restarts: int, seed: int, workers: int) -> List[Action]:
    best_actions: List[Action] = []
    best_score = -1

    if workers <= 1 or restarts <= 1:
        for restart in range(restarts):
            candidate = run_single_restart(dataset, restart, iterations, seed)
            score, _, _, _ = evaluate_actions(candidate, dataset)
            if score > best_score or (score == best_score and (not best_actions or len(candidate) < len(best_actions))):
                best_actions = candidate
                best_score = score
        return best_actions

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_single_restart, dataset, restart, iterations, seed)
            for restart in range(restarts)
        ]

        for future in futures:
            candidate = future.result()
            score, _, _, _ = evaluate_actions(candidate, dataset)
            if score > best_score or (score == best_score and (not best_actions or len(candidate) < len(best_actions))):
                best_actions = candidate
                best_score = score

    return best_actions


def run_solver(dataset_id: int, iterations: int, restarts: int, workers: int, write_solution: bool) -> List[Action]:
    if dataset_id != 2:
        raise ValueError("solver_dataset_2 ne supporte que --dataset 2")
    if workers < 1:
        raise ValueError("--workers doit etre >= 1")
    if restarts < 1:
        raise ValueError("--restarts doit etre >= 1")
    if iterations < 0:
        raise ValueError("--iterations doit etre >= 0")

    cpu_count = os.cpu_count() or 1
    worker_count = min(workers, cpu_count, restarts)
    run_seed = random.SystemRandom().randint(1, 2_147_483_647)

    dataset_path = ROOT_DIR / "datasets" / "dataset_2.json"
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
    print("Dataset: dataset_2.json")
    print(f"Actions: {len(best_actions)}")
    print(f"Valid  : {is_valid}")
    print(f"Score  : {score:_}")
    print(f"Message: {message}")
    print(f"Temps  : {elapsed:.2f}s")
    print(f"Seed run: {run_seed}")
    print(f"CPU workers utilises: {worker_count}")
    print("---------------------------------")

    if write_solution:
        out = ROOT_DIR / "solutions" / "solution_2.txt"
        out.write_text(solution_txt + "\n", encoding="utf-8")
        print(f"Solution ecrite dans: {out}")

    return best_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy + Monte Carlo optimizer for dataset_2")
    parser.add_argument("--dataset", type=int, default=2, choices=[2], help="Dataset number (must be 2)")
    parser.add_argument("--iterations", type=int, default=5000, help="MC iterations per restart")
    parser.add_argument("--restarts", type=int, default=6, help="Number of random restarts")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de processus CPU (1 = desactive le parallelisme)")
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_2.txt")
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

