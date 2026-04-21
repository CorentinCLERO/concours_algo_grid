import argparse
import json
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_solution

Grid = List[List[int]]
Candidate = Tuple[int, int, int, int, int, int]  # area, x1, y1, x2, y2, color
RectAction = Tuple[int, int, int, int, int]
JokerAction = Tuple[str, int, int, int, int, int]
Action = RectAction | JokerAction
UNUSED_COMPAT_HELP = "Unused compatibility parameter"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_text(actions: List[Action]) -> str:
    out = []
    for action in actions:
        if len(action) == 5:
            x1, y1, x2, y2, c = action
            out.append(f"RECT {x1} {y1} {x2} {y2} {c}")
        else:
            _, x1, y1, x2, y2, _ = action
            out.append(f"JOKER {x1} {y1} {x2} {y2}")
    return "\n".join(out)


def apply(grid: Grid, target: Grid, action: Action) -> None:
    if len(action) == 6:
        _, x1, y1, x2, y2, _ = action
        for y in range(y1, y2 + 1):
            row = grid[y]
            targ = target[y]
            for x in range(x1, x2 + 1):
                row[x] = targ[x]
        return

    x1, y1, x2, y2, c = action
    for y in range(y1, y2 + 1):
        row = grid[y]
        for x in range(x1, x2 + 1):
            row[x] = c


def update_best_from_histogram(heights: List[int], y: int, best_box: Optional[Tuple[int, int, int, int]], best_area: int) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    stack: List[Tuple[int, int]] = []
    w = len(heights)
    for x in range(w + 1):
        cur = heights[x] if x < w else 0
        start = x
        while stack and stack[-1][1] >= cur:
            idx, height = stack.pop()
            area = height * (x - idx)
            if area > best_area:
                best_area = area
                best_box = (idx, y - height + 1, x - 1, y)
            start = idx
        stack.append((start, cur))
    return best_box, best_area


def largest_rectangle(mask: Grid) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    h = len(mask)
    w = len(mask[0])
    heights = [0] * w
    best_box = None
    best_area = 0

    for y in range(h):
        row = mask[y]
        for x in range(w):
            heights[x] = heights[x] + 1 if row[x] else 0
        best_box, best_area = update_best_from_histogram(heights, y, best_box, best_area)

    return best_box, best_area


def erase_box(mask: Grid, x1: int, y1: int, x2: int, y2: int) -> None:
    for y in range(y1, y2 + 1):
        row = mask[y]
        for x in range(x1, x2 + 1):
            row[x] = 0


def build_exact_rectangles(target: Grid) -> List[Candidate]:
    rects: List[Candidate] = []
    colors = sorted({cell for row in target for cell in row if cell != 0})

    for color in colors:
        mask = [[1 if cell == color else 0 for cell in row] for row in target]
        while True:
            box, area = largest_rectangle(mask)
            if box is None or area <= 0:
                break
            x1, y1, x2, y2 = box
            rects.append((area, x1, y1, x2, y2, color))
            erase_box(mask, x1, y1, x2, y2)

    rects.sort(key=lambda r: (-r[0], r[2], r[1], r[4], r[3], r[5]))
    return rects


def build_mask_from_rects(h: int, w: int, rects: List[Candidate]) -> Grid:
    mask = [[0] * w for _ in range(h)]
    for _, x1, y1, x2, y2, _ in rects:
        for y in range(y1, y2 + 1):
            row = mask[y]
            for x in range(x1, x2 + 1):
                row[x] = 1
    return mask


def intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def best_subarray_at_most(values: List[int], limit: int) -> Tuple[int, Tuple[int, int]]:
    prefix = 0
    best_gain = 0
    best_interval = (0, -1)
    window: deque[Tuple[int, int]] = deque([(0, 0)])

    for x in range(1, len(values) + 1):
        prefix += values[x - 1]
        while window and window[0][0] < x - limit:
            window.popleft()
        if window:
            gain = prefix - window[0][1]
            if gain > best_gain:
                best_gain = gain
                best_interval = (window[0][0], x - 1)
        while window and window[-1][1] >= prefix:
            window.pop()
        window.append((x, prefix))

    return best_gain, best_interval


def joker_for_row_range(residual: Grid, y1: int, y2: int, max_area: int, col: List[int]) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    height = y2 - y1 + 1
    max_width = max_area // height
    if max_width <= 0:
        return 0, None

    for x in range(len(col)):
        col[x] += residual[y2][x]

    gain, (x1, x2) = best_subarray_at_most(col, max_width)
    if gain <= 0:
        return 0, None
    return gain, (x1, y1, x2, y2)


def best_joker(target: Grid, covered: Grid, max_area: int) -> Tuple[Optional[Action], int]:
    h = len(target)
    w = len(target[0])
    residual = [[1 if target[y][x] != 0 and not covered[y][x] else 0 for x in range(w)] for y in range(h)]

    best_gain = 0
    best_box: Optional[Tuple[int, int, int, int]] = None
    col = [0] * w

    for y1 in range(h):
        for x in range(w):
            col[x] = 0
        for y2 in range(y1, min(h, y1 + max_area)):
            gain, box = joker_for_row_range(residual, y1, y2, max_area, col)
            if box is not None and gain > best_gain:
                best_gain = gain
                best_box = box

    if best_box is None or best_gain <= 0:
        return None, 0
    x1, y1, x2, y2 = best_box
    joker_action: JokerAction = ("JOKER", x1, y1, x2, y2, -1)
    return joker_action, best_gain


def choose_solution(target: Grid, max_actions: int, max_jokers: int) -> List[Action]:
    exact_rects = build_exact_rectangles(target)
    if not exact_rects:
        return []

    exact_only_candidates = exact_rects[: min(max_actions, len(exact_rects))]
    exact_only: List[Action] = [(r[1], r[2], r[3], r[4], r[5]) for r in exact_only_candidates]
    exact_only_gain = sum(r[0] for r in exact_only_candidates)

    if max_jokers <= 0 or len(exact_rects) <= max_actions:
        return exact_only

    big_rects = [r for r in exact_rects if r[0] > 4]
    small_rects = [r for r in exact_rects if r[0] == 4]

    if len(big_rects) >= max_actions:
        return exact_only

    h = len(target)
    w = len(target[0])
    covered = build_mask_from_rects(h, w, big_rects)
    joker, joker_gain = best_joker(target, covered, 400)

    if joker is None or joker_gain <= 4:
        return exact_only

    joker_box = joker[1], joker[2], joker[3], joker[4]
    need = max_actions - 1 - len(big_rects)
    if need < 0:
        need = 0

    outside = [r for r in small_rects if not intersect((r[1], r[2], r[3], r[4]), joker_box)]
    inside = [r for r in small_rects if intersect((r[1], r[2], r[3], r[4]), joker_box)]

    chosen_small: List[Action] = [(r[1], r[2], r[3], r[4], r[5]) for r in outside[:need]]
    if len(chosen_small) < need:
        chosen_small.extend((r[1], r[2], r[3], r[4], r[5]) for r in inside[: need - len(chosen_small)])

    joker_candidate_gain = sum(r[0] for r in big_rects) + 4 * len(chosen_small) + joker_gain
    if exact_only_gain >= joker_candidate_gain:
        return exact_only

    return [(r[1], r[2], r[3], r[4], r[5]) for r in big_rects] + chosen_small + [joker]


def build_solution(dataset: dict, max_actions: int, max_jokers: int) -> List[Action]:
    target: Grid = dataset["grid"]
    return choose_solution(target, max_actions, max_jokers)


def simulate_gain_rect(grid: Grid, target: Grid, x1: int, y1: int, x2: int, y2: int, color: int) -> int:
    gain = 0
    for y in range(y1, y2 + 1):
        row = grid[y]
        targ = target[y]
        for x in range(x1, x2 + 1):
            before = 1 if row[x] == targ[x] else 0
            after = 1 if color == targ[x] else 0
            gain += after - before
    return gain


def simulate_gain_joker(grid: Grid, target: Grid, x1: int, y1: int, x2: int, y2: int) -> int:
    gain = 0
    for y in range(y1, y2 + 1):
        row = grid[y]
        targ = target[y]
        for x in range(x1, x2 + 1):
            gain += 1 if row[x] != targ[x] else 0
    return gain


def simulate_gain(grid: Grid, target: Grid, action: Action) -> int:
    if len(action) == 6:
        _, x1, y1, x2, y2, _ = action
        return simulate_gain_joker(grid, target, x1, y1, x2, y2)
    x1, y1, x2, y2, color = action
    return simulate_gain_rect(grid, target, x1, y1, x2, y2, color)


def score_actions(actions: List[Action], dataset: dict) -> Tuple[int, bool, str]:
    solution_txt = to_text(actions)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    return test_solution.get_solution_score(solution_txt, dataset_txt)


def run_box(target: Grid, x: int, y: int, color: int) -> Tuple[int, int, int, int]:
    h = len(target)
    w = len(target[0])
    left = x
    while left > 0 and target[y][left - 1] == color:
        left -= 1
    right = x
    while right + 1 < w and target[y][right + 1] == color:
        right += 1

    top = y
    while top > 0 and all(target[top - 1][xx] == color for xx in range(left, right + 1)):
        top -= 1
    bottom = y
    while bottom + 1 < h and all(target[bottom + 1][xx] == color for xx in range(left, right + 1)):
        bottom += 1
    return left, top, right, bottom


def random_candidates(target: Grid, x: int, y: int, rng: random.Random) -> List[RectAction]:
    h = len(target)
    w = len(target[0])
    color = target[y][x]
    out: List[RectAction] = []
    seen = set()

    def add(x1: int, y1: int, x2: int, y2: int, c: int) -> None:
        if x1 > x2 or y1 > y2:
            return
        key = (x1, y1, x2, y2, c)
        if key not in seen:
            seen.add(key)
            out.append(key)

    add(x, y, x, y, color)

    # Exact local rectangle in target for the seed color.
    if color != 0:
        x1, y1, x2, y2 = run_box(target, x, y, color)
        add(x1, y1, x2, y2, color)

    # Random sampled boxes around the seed for both paint and cleanup.
    widths = (2, 4, 6, 8, 12, 16, 24, 32)
    heights = (2, 4, 6, 8, 12, 16, 24, 32)
    for _ in range(12):
        ww = rng.choice(widths)
        hh = rng.choice(heights)
        x1 = max(0, min(w - ww, x - rng.randint(0, ww - 1)))
        y1 = max(0, min(h - hh, y - rng.randint(0, hh - 1)))
        x2 = x1 + ww - 1
        y2 = y1 + hh - 1
        add(x1, y1, x2, y2, color)
        if color != 0:
            add(x1, y1, x2, y2, 0)

    return out


def best_joker_mismatch(grid: Grid, target: Grid, max_area: int) -> Tuple[Optional[JokerAction], int]:
    h = len(target)
    w = len(target[0])
    residual = [[1 if grid[y][x] != target[y][x] else 0 for x in range(w)] for y in range(h)]

    best_gain = 0
    best_box: Optional[Tuple[int, int, int, int]] = None
    col = [0] * w

    for y1 in range(h):
        for x in range(w):
            col[x] = 0
        for y2 in range(y1, min(h, y1 + max_area)):
            gain, box = joker_for_row_range(residual, y1, y2, max_area, col)
            if box is not None and gain > best_gain:
                best_gain = gain
                best_box = box

    if best_box is None or best_gain <= 0:
        return None, 0
    x1, y1, x2, y2 = best_box
    return ("JOKER", x1, y1, x2, y2, -1), best_gain


def select_best_rect_action(grid: Grid, target: Grid, mismatches: List[Tuple[int, int]], seeds_per_step: int, rng: random.Random) -> Tuple[Optional[RectAction], int]:
    best_gain = 0
    best_action: Optional[RectAction] = None
    sample_size = min(len(mismatches), seeds_per_step)
    seeds = rng.sample(mismatches, sample_size) if sample_size < len(mismatches) else mismatches

    for x, y in seeds:
        for cand in random_candidates(target, x, y, rng):
            gain = simulate_gain(grid, target, cand)
            if gain > best_gain:
                best_gain = gain
                best_action = cand
    return best_action, best_gain


def try_promote_joker(best_gain: int, jokers_left: int, grid: Grid, target: Grid, max_joker_size: int, min_joker_gain: int) -> Tuple[Optional[Action], int]:
    if jokers_left <= 0:
        return None, best_gain
    joker, joker_gain = best_joker_mismatch(grid, target, max_joker_size)
    if joker is not None and joker_gain >= max(min_joker_gain, best_gain + 2):
        return joker, joker_gain
    return None, best_gain


def build_solution_search(dataset: dict, seed: int, iterations: int) -> List[Action]:
    rng = random.Random(seed)
    target: Grid = dataset["grid"]
    h = len(target)
    w = len(target[0])
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    grid = [[0] * w for _ in range(h)]
    actions: List[Action] = []
    jokers_left = max_jokers

    seeds_per_step = max(40, min(180, 30 + iterations // 600))
    min_joker_gain = 8

    for _ in range(max_actions):
        mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
        if not mismatches:
            break

        rect_action, best_gain = select_best_rect_action(grid, target, mismatches, seeds_per_step, rng)
        best_action: Optional[Action] = rect_action

        joker_action, joker_gain = try_promote_joker(best_gain, jokers_left, grid, target, max_joker_size, min_joker_gain)
        if joker_action is not None:
            best_action = joker_action
            best_gain = joker_gain

        if best_action is None or best_gain <= 0:
            break

        apply(grid, target, best_action)
        actions.append(best_action)
        if len(best_action) == 6:
            jokers_left -= 1

    return actions


def action_box(action: Action) -> Tuple[int, int, int, int]:
    if len(action) == 6:
        _, x1, y1, x2, y2, _ = action
        return x1, y1, x2, y2
    x1, y1, x2, y2, _ = action
    return x1, y1, x2, y2


def mark_last_writer(last_writer: Grid, actions: List[Action]) -> None:
    for idx, action in enumerate(actions):
        x1, y1, x2, y2 = action_box(action)
        for y in range(y1, y2 + 1):
            row = last_writer[y]
            for x in range(x1, x2 + 1):
                row[x] = idx


def mark_kept_actions(last_writer: Grid, actions_count: int) -> List[bool]:
    h = len(last_writer)
    w = len(last_writer[0])
    keep = [False] * actions_count
    for y in range(h):
        for x in range(w):
            idx = last_writer[y][x]
            if idx >= 0:
                keep[idx] = True
    return keep


def compact_actions_by_last_writer(actions: List[Action], dataset: dict) -> List[Action]:
    target: Grid = dataset["grid"]
    h = len(target)
    w = len(target[0])
    if not actions:
        return []

    last_writer = [[-1] * w for _ in range(h)]
    mark_last_writer(last_writer, actions)
    keep = mark_kept_actions(last_writer, len(actions))

    return [action for i, action in enumerate(actions) if keep[i]]


def replay_actions(actions: List[Action], dataset: dict) -> Tuple[Grid, int]:
    target: Grid = dataset["grid"]
    h = len(target)
    w = len(target[0])
    grid = [[0] * w for _ in range(h)]
    jokers_used = 0
    for action in actions:
        apply(grid, target, action)
        if len(action) == 6:
            jokers_used += 1
    return grid, jokers_used


def refill_after_compaction(actions: List[Action], dataset: dict, seed: int, iterations: int) -> List[Action]:
    target: Grid = dataset["grid"]
    h = len(target)
    w = len(target[0])
    max_actions = dataset["maxActions"]
    max_jokers = dataset["maxJokers"]
    max_joker_size = dataset["maxJokerSize"]

    if len(actions) >= max_actions:
        return actions

    rng = random.Random(seed)
    current = list(actions)
    grid, jokers_used = replay_actions(current, dataset)
    jokers_left = max(0, max_jokers - jokers_used)

    seeds_per_step = max(24, min(90, 16 + iterations // 900))
    min_joker_gain = 6

    while len(current) < max_actions:
        mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
        if not mismatches:
            break

        rect_action, best_gain = select_best_rect_action(grid, target, mismatches, seeds_per_step, rng)
        best_action: Optional[Action] = rect_action

        joker_action, joker_gain = try_promote_joker(best_gain, jokers_left, grid, target, max_joker_size, min_joker_gain)
        if joker_action is not None:
            best_action = joker_action
            best_gain = joker_gain

        if best_action is None or best_gain <= 0:
            break

        apply(grid, target, best_action)
        current.append(best_action)
        if len(best_action) == 6:
            jokers_left -= 1

    return current


def improve_candidate(candidate: List[Action], dataset: dict, seed: int, iterations: int) -> List[Action]:
    compacted = compact_actions_by_last_writer(candidate, dataset)
    if len(compacted) >= dataset["maxActions"]:
        return compacted

    # Refill the freed slots and compact once again to remove new overwritten moves.
    refilled = refill_after_compaction(compacted, dataset, seed, iterations)
    return compact_actions_by_last_writer(refilled, dataset)


def best_of_variants(variants: List[List[Action]], dataset: dict) -> List[Action]:
    best: List[Action] = []
    best_score = -1
    for actions in variants:
        score, ok, _ = score_actions(actions, dataset)
        if not ok:
            continue
        if score > best_score or (score == best_score and (not best or len(actions) < len(best))):
            best = actions
            best_score = score
    return best


def run_once(dataset: dict, restart: int, base_seed: int, iterations: int) -> List[Action]:
    seed = base_seed + restart * 7919
    candidate = build_solution_search(dataset, seed, iterations)
    improved = improve_candidate(candidate, dataset, seed ^ 0xA5A5A5A5, iterations)
    exact = build_solution(dataset, dataset["maxActions"], dataset["maxJokers"])
    best = best_of_variants([candidate, improved, exact], dataset)
    return best if best else exact


def is_better(candidate: List[Action], current_best: List[Action], dataset: dict) -> bool:
    score_c, ok_c, _ = score_actions(candidate, dataset)
    score_b, ok_b, _ = score_actions(current_best, dataset)
    if not ok_c:
        return False
    if not ok_b:
        return True
    if score_c > score_b:
        return True
    return score_c == score_b and len(candidate) < len(current_best)


def find_best_serial(dataset: dict, restarts: int, iterations: int, base_seed: int, best: List[Action]) -> List[Action]:
    for restart in range(max(1, restarts)):
        cand = run_once(dataset, restart, base_seed, iterations)
        if is_better(cand, best, dataset):
            best = cand
    return best


def find_best_parallel(dataset: dict, restarts: int, iterations: int, workers: int, base_seed: int, best: List[Action]) -> List[Action]:
    cpu = os.cpu_count() or 1
    max_workers = min(cpu, workers, restarts)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_once, dataset, r, base_seed, iterations) for r in range(restarts)]
        for future in futures:
            cand = future.result()
            if is_better(cand, best, dataset):
                best = cand
    return best


def find_best(dataset: dict, restarts: int, workers: int, iterations: int) -> List[Action]:
    best = build_solution(dataset, dataset["maxActions"], dataset["maxJokers"])
    base_seed = random.SystemRandom().randint(1, 2_147_483_647)

    if restarts <= 1 or workers <= 1:
        return find_best_serial(dataset, restarts, iterations, base_seed, best)
    return find_best_parallel(dataset, restarts, iterations, workers, base_seed, best)


def run_solver(dataset_id: int, iterations: int, restarts: int, workers: int, write_solution: bool) -> List[Action]:
    dataset_path = ROOT_DIR / "datasets" / f"dataset_{dataset_id}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    dataset = load_dataset(dataset_path)
    started = time.time()
    best = find_best(dataset, restarts, workers, iterations)
    elapsed = time.time() - started

    solution_txt = to_text(best)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    score, ok, msg = test_solution.get_solution_score(solution_txt, dataset_txt)

    print("---------------------------------")
    print(f"Dataset: dataset_{dataset_id}.json")
    print(f"Actions: {len(best)}")
    print(f"Valid  : {ok}")
    print(f"Score  : {score:_}")
    print(f"Message: {msg}")
    print(f"Temps  : {elapsed:.2f}s")
    print("Approche: exact cover + recherche stochastique parallele")
    print("---------------------------------")

    if write_solution:
        out = ROOT_DIR / "solutions" / f"solution_{dataset_id}.txt"
        out.write_text(solution_txt + "\n", encoding="utf-8")
        print(f"Solution ecrite dans: {out}")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact rectangle cover optimizer for dataset 4")
    parser.add_argument("--dataset", type=int, default=4, help="Dataset number (default: 4)")
    parser.add_argument("--iterations", type=int, default=50000, help=UNUSED_COMPAT_HELP)
    parser.add_argument("--restarts", type=int, default=14, help=UNUSED_COMPAT_HELP)
    parser.add_argument("--workers", type=int, default=14, help=UNUSED_COMPAT_HELP)
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_<dataset>.txt")
    args = parser.parse_args()

    run_solver(args.dataset, args.iterations, args.restarts, args.workers, args.write)


if __name__ == "__main__":
    main()

