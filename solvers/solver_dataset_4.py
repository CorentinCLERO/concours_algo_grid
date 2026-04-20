import argparse
import json
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import test_solution

Rect = Tuple[int, int, int, int, int]  # x1, y1, x2, y2, color
Grid = List[List[int]]
Component = Tuple[int, int, int, int]


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_text(actions: List[Rect]) -> str:
    out = []
    for action in actions:
        if len(action) == 5:
            x1, y1, x2, y2, c = action
            out.append(f"RECT {x1} {y1} {x2} {y2} {c}")
        else:
            _, x1, y1, x2, y2, _ = action
            out.append(f"JOKER {x1} {y1} {x2} {y2}")
    return "\n".join(out)


def apply(grid: Grid, target: Grid, action) -> None:
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


def simulate_gain(grid: Grid, target: Grid, action: Rect) -> int:
    kind, x1, y1, x2, y2, c = action
    gain = 0
    for y in range(y1, y2 + 1):
        row = grid[y]
        targ = target[y]
        for x in range(x1, x2 + 1):
            before = row[x] == targ[x]
            after_val = targ[x] if kind == "JOKER" else c
            after = after_val == targ[x]
            gain += int(after) - int(before)
    return gain


def build_comp_boxes(target: Grid) -> Dict[int, List[Component]]:
    h, w = len(target), len(target[0])
    seen = [[False] * w for _ in range(h)]
    by_color: Dict[int, List[Component]] = {}

    for y in range(h):
        for x in range(w):
            c = target[y][x]
            if c == 0 or seen[y][x]:
                continue
            q = deque([(x, y)])
            seen[y][x] = True
            x1 = x2 = x
            y1 = y2 = y
            while q:
                cx, cy = q.popleft()
                if cx < x1:
                    x1 = cx
                if cx > x2:
                    x2 = cx
                if cy < y1:
                    y1 = cy
                if cy > y2:
                    y2 = cy
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and target[ny][nx] == c:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            by_color.setdefault(c, []).append((x1, y1, x2, y2))

    return by_color


def exact_run_box(target: Grid, x: int, y: int) -> Tuple[int, int, int, int, int]:
    c = target[y][x]
    h, w = len(target), len(target[0])
    left = x
    while left > 0 and target[y][left - 1] == c:
        left -= 1
    right = x
    while right + 1 < w and target[y][right + 1] == c:
        right += 1
    top = y
    while top > 0 and all(target[top - 1][xx] == c for xx in range(left, right + 1)):
        top -= 1
    bottom = y
    while bottom + 1 < h and all(target[bottom + 1][xx] == c for xx in range(left, right + 1)):
        bottom += 1
    return left, top, right, bottom, c


def candidate_rectangles(target: Grid, comp_boxes: Dict[int, List[Component]], x: int, y: int, rng: random.Random) -> List[Rect]:
    c = target[y][x]
    if c == 0:
        return []

    h, w = len(target), len(target[0])
    run = exact_run_box(target, x, y)

    # Try to match the component box containing the seed.
    comp_box = None
    for box in comp_boxes.get(c, []):
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2:
            comp_box = (x1, y1, x2, y2, c)
            break

    bases: List[Tuple[int, int, int, int, int]] = [run]
    if comp_box is not None:
        bases.append(comp_box)

    candidates: List[Rect] = []
    for bx1, by1, bx2, by2, color in bases:
        candidates.append((bx1, by1, bx2, by2, color))
        # Reduced to 8 variations for better performance
        for _ in range(8):
            dx1 = rng.randint(0, 15)
            dy1 = rng.randint(0, 15)
            dx2 = rng.randint(0, 15)
            dy2 = rng.randint(0, 15)
            x1 = max(0, bx1 - dx1)
            y1 = max(0, by1 - dy1)
            x2 = min(w - 1, bx2 + dx2)
            y2 = min(h - 1, by2 + dy2)
            candidates.append((x1, y1, x2, y2, color))

    # Tiny joker candidates on small mismatched regions.
    area = (run[2] - run[0] + 1) * (run[3] - run[1] + 1)
    if area <= 400:
        candidates.append((run[0], run[1], run[2], run[3], c))

    # Dedup.
    seen = set()
    out = []
    for a in candidates:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def choose_best_action(grid: Grid, target: Grid, comp_boxes: Dict[int, List[Component]], rng: random.Random, max_jokers: int, jokers_used: int) -> Tuple[Rect | None, int]:
    h, w = len(target), len(target[0])
    mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
    if not mismatches:
        return None, 0

    rng.shuffle(mismatches)
    # Increased from 80 to 200 for better exploration
    seeds = mismatches[: min(200, len(mismatches))]

    best_action: Rect | None = None
    best_gain = 0

    for x, y in seeds:
        for cand in candidate_rectangles(target, comp_boxes, x, y, rng):
            gain = simulate_gain(grid, target, ("RECT", *cand[:4], cand[4]))
            if gain > best_gain:
                best_gain = gain
                best_action = (cand[0], cand[1], cand[2], cand[3], cand[4])

        # We don't use Joker here anymore, save it for the repair phase

    return best_action, best_gain


def count_mismatches_in_box(grid: Grid, target: Grid, x1: int, y1: int, x2: int, y2: int) -> int:
    """Count how many pixels in the box don't match target."""
    count = 0
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            if grid[y][x] != target[y][x]:
                count += 1
    return count


def find_dense_regions(grid: Grid, target: Grid, rng: random.Random) -> List[Tuple[int, int]]:
    """Find clusters of mismatched pixels and return representative centers."""
    h, w = len(target), len(target[0])
    mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
    if not mismatches:
        return []

    # Use density-based clustering: divide grid into regions and find dense ones.
    regions = {}
    region_size = 16
    for x, y in mismatches:
        rx = x // region_size
        ry = y // region_size
        key = (rx, ry)
        regions[key] = regions.get(key, 0) + 1

    # Sort by density (most dense first) and return one seed per region.
    dense_regions = sorted(regions.items(), key=lambda kv: -kv[1])
    seeds = []
    for (rx, ry), count in dense_regions:
        # Pick a random mismatch in this region.
        candidates = [p for p in mismatches if p[0] // region_size == rx and p[1] // region_size == ry]
        if candidates:
            seeds.append(rng.choice(candidates))
    return seeds


def optimize_action_coverage(grid: Grid, target: Grid, actions: List[Rect], max_actions: int, rng: random.Random) -> List[Rect]:
    """Try to improve pixel coverage by strategic action selection from dense mismatch regions."""
    h, w = len(target), len(target[0])

    # Apply all actions to get final state.
    test_grid = [[0 for _ in range(w)] for _ in range(h)]
    for action in actions:
        apply(test_grid, target, action)

    # Find remaining mismatches.
    remaining = [(x, y) for y in range(h) for x in range(w) if test_grid[y][x] != target[y][x]]
    if not remaining or len(actions) >= max_actions:
        return actions

    # Build a set of action signatures already in list (to avoid duplicates).
    action_set = set(actions)

    # Sample candidate replacements from high-value regions. Use empty comp_boxes for speed.
    candidates = []
    sample_size = min(40, len(remaining))
    for seed_x, seed_y in rng.sample(remaining, sample_size):
        if target[seed_y][seed_x] == 0:
            continue
        for cand in candidate_rectangles(target, {}, seed_x, seed_y, rng):
            if cand not in action_set:
                gain = simulate_gain(test_grid, target, ("RECT", *cand[:4], cand[4]))
                if gain > 0:
                    candidates.append((cand, gain))

    # Sort by gain and try to add the best ones.
    candidates.sort(key=lambda item: -item[1])
    for cand, _ in candidates[:min(100, len(candidates))]:
        if len(actions) >= max_actions:
            break
        if cand not in action_set:
            actions.append(cand)
            action_set.add(cand)
            apply(test_grid, target, cand)

    return actions



def build_solution(dataset: dict, rng: random.Random, max_actions: int, max_jokers: int, iterations: int = 100000) -> List[Rect]:
    target: Grid = dataset["grid"]
    h, w = len(target), len(target[0])
    comp_boxes = build_comp_boxes(target)
    grid = [[0 for _ in range(w)] for _ in range(h)]
    actions: List[Rect] = [(0, 0, w - 1, h - 1, 0)]
    jokers_used = 0

    # Greedy gain-based construction.
    while len(actions) < max_actions - 1:
        action, gain = choose_best_action(grid, target, comp_boxes, rng, max_jokers, jokers_used)
        if action is None or gain < 0:  # Accept neutral gains too
            break
        actions.append(action)
        apply(grid, target, action)
        if len(action) == 6 and action[0] == "JOKER":
            jokers_used += 1

    # Repair phase: aggressive filling of remaining budget.
    while len(actions) < max_actions - 1:
        mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
        if not mismatches:
            break

        rng.shuffle(mismatches)
        found_any = False

        # Try to add repairs for up to 30 mismatches in this round.
        for x, y in mismatches[:min(200, len(mismatches))]:
            if len(actions) >= max_actions - 1:
                break

            c = target[y][x]
            if c == 0:
                continue

            # Use exact run box for quick repair.
            x1, y1, x2, y2, color = exact_run_box(target, x, y)
            cand = (x1, y1, x2, y2, color)

            # Check if we haven't already added this exact action.
            if cand not in actions:
                gain = simulate_gain(grid, target, cand)
                if gain > 0:
                    actions.append(cand)
                    apply(grid, target, cand)
                    found_any = True

        if not found_any:
            # If exact_run_box didn't yield positive gain, try a 1x1 box
            for x, y in mismatches[:min(200, len(mismatches))]:
                if len(actions) >= max_actions - 1:
                    break
                c = target[y][x]
                if c == 0:
                    continue
                cand = (x, y, x, y, c)
                if cand not in actions:
                    gain = simulate_gain(grid, target, cand)
                    if gain > 0:
                        actions.append(cand)
                        apply(grid, target, cand)
                        found_any = True
            if not found_any:
                break

    # If we still have a Joker, let's try placing it to cover remaining mismatches
    if jokers_used < max_jokers and len(actions) < max_actions:
        mismatches = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != target[y][x]]
        if mismatches:
            # Center a joker around the densest area of mismatches
            best_joker = None
            best_gain = 0
            for x, y in mismatches[:min(100, len(mismatches))]:
                for w_j in range(1, min(w, 400)+1):
                    h_j = min(h, 400 // w_j)
                    cx = max(0, min(x - w_j//2, w - w_j))
                    cy = max(0, min(y - h_j//2, h - h_j))
                    cand = ("JOKER", cx, cy, cx + w_j - 1, cy + h_j - 1, -1)
                    gain = simulate_gain(grid, target, cand)
                    if gain > best_gain:
                        best_gain = gain
                        best_joker = cand
            if best_joker and best_gain > 0:
                actions.append(best_joker)
                apply(grid, target, best_joker)

    return actions


def run_once(dataset: dict, restart: int, seed: int, iterations: int) -> List[Rect]:
    rng = random.Random(seed + restart * 1013)
    # Slightly vary the randomness between restarts.
    return build_solution(dataset, rng, max_actions=dataset["maxActions"], max_jokers=dataset["maxJokers"], iterations=iterations)


def find_best(dataset: dict, restarts: int, seed: int, workers: int, iterations: int) -> List[Rect]:
    best: List[Rect] = []
    best_score = -1

    def consider(candidate: List[Rect]) -> None:
        nonlocal best, best_score
        s, ok, _ = test_solution.get_solution_score(to_text(candidate), json.dumps(dataset, ensure_ascii=False))
        if ok and (s > best_score or (s == best_score and (not best or len(candidate) < len(best)))):
            best = candidate
            best_score = s

    if workers <= 1 or restarts <= 1:
        for restart in range(restarts):
            consider(run_once(dataset, restart, seed, iterations))
        return best

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_once, dataset, restart, seed, iterations) for restart in range(restarts)]
        for future in futures:
            consider(future.result())

    return best


def run_solver(dataset_id: int, iterations: int, restarts: int, workers: int, write_solution: bool) -> List[Rect]:
    if workers < 1:
        raise ValueError("--workers doit etre >= 1")
    if restarts < 1:
        raise ValueError("--restarts doit etre >= 1")

    cpu_count = os.cpu_count() or 1
    worker_count = min(workers, cpu_count, restarts)
    seed = random.SystemRandom().randint(1, 2_147_483_647)

    dataset_path = ROOT_DIR / "datasets" / f"dataset_{dataset_id}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    dataset = load_dataset(dataset_path)
    started = time.time()
    best = find_best(dataset, restarts, seed, worker_count, iterations)
    elapsed = time.time() - started

    solution_txt = to_text(best)
    dataset_txt = json.dumps(dataset, ensure_ascii=False)
    s, ok, msg = test_solution.get_solution_score(solution_txt, dataset_txt)

    print("---------------------------------")
    print(f"Dataset: dataset_{dataset_id}.json")
    print(f"Actions: {len(best)}")
    print(f"Valid  : {ok}")
    print(f"Score  : {s:_}")
    print(f"Message: {msg}")
    print(f"Temps  : {elapsed:.2f}s")
    print(f"Seed run: {seed}")
    print(f"CPU workers utilises: {worker_count}")
    print("---------------------------------")

    if write_solution:
        out = ROOT_DIR / "solutions" / f"solution_{dataset_id}.txt"
        out.write_text(solution_txt + "\n", encoding="utf-8")
        print(f"Solution ecrite dans: {out}")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy rectangle optimizer for dataset_4")
    parser.add_argument("--dataset", type=int, default=4, help="Dataset number (default: 4)")
    parser.add_argument("--iterations", type=int, default=300000, help="Repair phase iterations parameter")
    parser.add_argument("--restarts", type=int, default=14, help="Number of random restarts")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de processus CPU (1 = desactive le parallelisme)")
    parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_<dataset>.txt")
    args = parser.parse_args()

    run_solver(args.dataset, args.iterations, args.restarts, args.workers, args.write)


if __name__ == "__main__":
    main()

