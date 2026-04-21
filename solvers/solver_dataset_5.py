from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

import test_solution

Grid = List[List[int]]
RectAction = Tuple[int, int, int, int, int]
JokerAction = Tuple[str, int, int, int, int, int]
Action = RectAction | JokerAction

STRATEGIES = ("savings", "density", "compact", "balanced", "color_bias")


@dataclass(frozen=True)
class JokerCandidate:
	savings: int
	cell_count: int
	area: int
	color: int
	bbox: Tuple[int, int, int, int]
	tie: float


class NullProgressSink:
	def start(self, total: int, desc: str) -> None:
		return

	def set_total(self, total: int) -> None:
		return

	def advance(self, delta: int = 1, desc: Optional[str] = None) -> None:
		return

	def close(self, desc: Optional[str] = None) -> None:
		return


class LocalProgressSink:
	def __init__(self, position: int, desc: str, enabled: bool) -> None:
		self.enabled = enabled
		self.started = False
		self.bar = tqdm(total=0, desc=desc, position=position, leave=False, dynamic_ncols=True, disable=not enabled)

	def start(self, total: int, desc: str) -> None:
		if self.enabled:
			if not self.started:
				self.bar.total = total
				self.started = True
			else:
				self.bar.total = (self.bar.total or 0) + total
			self.bar.set_description_str(desc)
			self.bar.refresh()

	def set_total(self, total: int) -> None:
		if self.enabled:
			self.bar.total = total
			self.bar.refresh()

	def advance(self, delta: int = 1, desc: Optional[str] = None) -> None:
		if self.enabled:
			if desc:
				self.bar.set_description_str(desc)
			self.bar.update(delta)

	def close(self, desc: Optional[str] = None) -> None:
		if self.enabled:
			if desc:
				self.bar.set_description_str(desc)
			self.bar.close()


class QueueProgressSink:
	def __init__(self, queue, task_id: int, enabled: bool) -> None:
		self.queue = queue
		self.task_id = task_id
		self.enabled = enabled

	def start(self, total: int, desc: str) -> None:
		if self.enabled:
			self.queue.put(("start", self.task_id, total, desc))

	def set_total(self, total: int) -> None:
		if self.enabled:
			self.queue.put(("total", self.task_id, total))

	def advance(self, delta: int = 1, desc: Optional[str] = None) -> None:
		if self.enabled:
			self.queue.put(("advance", self.task_id, delta, desc))

	def close(self, desc: Optional[str] = None) -> None:
		if self.enabled:
			self.queue.put(("close", self.task_id, desc))


def load_dataset(dataset_path: Path) -> dict:
	return json.loads(dataset_path.read_text(encoding="utf-8"))


def color_frequencies(grid: Grid) -> Counter:
	freq: Counter = Counter()
	for row in grid:
		freq.update(row)
	return freq


def action_to_text(action: Action) -> str:
	if len(action) == 5:
		x1, y1, x2, y2, color = action
		return f"RECT {x1} {y1} {x2} {y2} {color}"
	_, x1, y1, x2, y2, _ = action
	return f"JOKER {x1} {y1} {x2} {y2}"


def actions_to_solution_text(actions: Sequence[Action]) -> str:
	return "\n".join(action_to_text(action) for action in actions)


def apply_rect(grid: Grid, x1: int, y1: int, x2: int, y2: int, color: int) -> None:
	for y in range(y1, y2 + 1):
		row = grid[y]
		for x in range(x1, x2 + 1):
			row[x] = color


def apply_joker(grid: Grid, target: Grid, x1: int, y1: int, x2: int, y2: int) -> None:
	for y in range(y1, y2 + 1):
		row = grid[y]
		target_row = target[y]
		for x in range(x1, x2 + 1):
			row[x] = target_row[x]


def count_correct_cells(grid: Grid, target: Grid) -> int:
	correct = 0
	for y, row in enumerate(grid):
		target_row = target[y]
		for x, value in enumerate(row):
			if value == target_row[x]:
				correct += 1
	return correct


def evaluate_actions(actions: Sequence[Action], dataset: dict) -> Tuple[int, bool, str]:
	solution_txt = actions_to_solution_text(actions)
	dataset_txt = json.dumps(dataset, ensure_ascii=False)
	return test_solution.get_solution_score(solution_txt, dataset_txt)


def _component_row_run_count(cells: Sequence[Tuple[int, int]]) -> int:
	rows: dict[int, List[int]] = {}
	for x, y in cells:
		rows.setdefault(y, []).append(x)

	run_count = 0
	for xs in rows.values():
		xs.sort()
		prev = None
		for x in xs:
			if prev is None or x != prev + 1:
				run_count += 1
			prev = x
	return run_count


def _extract_component(
	target: Grid,
	seen: List[List[bool]],
	start_x: int,
	start_y: int,
	color: int,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
	h = len(target)
	w = len(target[0])
	queue = deque([(start_x, start_y)])
	seen[start_y][start_x] = True
	cells: List[Tuple[int, int]] = []
	min_x = max_x = start_x
	min_y = max_y = start_y

	while queue:
		cx, cy = queue.popleft()
		cells.append((cx, cy))
		if cx < min_x:
			min_x = cx
		if cx > max_x:
			max_x = cx
		if cy < min_y:
			min_y = cy
		if cy > max_y:
			max_y = cy

		for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
			nx = cx + dx
			ny = cy + dy
			if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and target[ny][nx] == color:
				seen[ny][nx] = True
				queue.append((nx, ny))

	return cells, (min_x, min_y, max_x, max_y)


def _candidate_from_component(
	cells: Sequence[Tuple[int, int]],
	bbox: Tuple[int, int, int, int],
	color: int,
	max_joker_size: int,
	rng: random.Random,
) -> Optional[JokerCandidate]:
	min_x, min_y, max_x, max_y = bbox
	area = (max_x - min_x + 1) * (max_y - min_y + 1)
	if area > max_joker_size:
		return None

	run_count = _component_row_run_count(cells)
	savings = run_count - 1
	if savings <= 0:
		return None

	return JokerCandidate(
		savings=savings,
		cell_count=len(cells),
		area=area,
		color=color,
		bbox=bbox,
		tie=rng.random(),
	)


def _component_candidates(target: Grid, base_color: int, max_joker_size: int, sink) -> List[JokerCandidate]:
	h = len(target)
	w = len(target[0])
	seen = [[False] * w for _ in range(h)]
	candidates: List[JokerCandidate] = []
	rng = random.Random((h << 16) ^ w ^ max_joker_size)

	sink.start(h, "analyse composantes")
	for y in range(h):
		sink.advance(1, desc="analyse composantes")
		row = target[y]
		for x in range(w):
			if seen[y][x] or row[x] == base_color:
				continue

			cells, bbox = _extract_component(target, seen, x, y, row[x])
			candidate = _candidate_from_component(cells, bbox, row[x], max_joker_size, rng)
			if candidate is not None:
				candidates.append(candidate)

	return candidates


def _candidate_sort_key(candidate: JokerCandidate, strategy: str) -> Tuple[float, float, float, float, float, float]:
	x1, y1, _, _ = candidate.bbox
	if strategy == "density":
		score = candidate.savings / candidate.area
	elif strategy == "compact":
		score = float(candidate.savings)
	elif strategy == "color_bias":
		score = float(candidate.savings)
	elif strategy == "balanced":
		score = candidate.savings + candidate.cell_count / candidate.area
	else:
		score = float(candidate.savings)

	return (
		score,
		float(candidate.savings),
		float(candidate.cell_count),
		float(-candidate.area),
		float(-candidate.color if strategy != "color_bias" else candidate.color),
		candidate.tie,
	)


def _select_jokers(candidates: List[JokerCandidate], max_jokers: int, strategy: str, sink) -> List[JokerCandidate]:
	sink.start(max(1, len(candidates)), "classement jokers")
	ranked = []
	for candidate in candidates:
		ranked.append((_candidate_sort_key(candidate, strategy), candidate))
		sink.advance(1, desc="classement jokers")

	ranked.sort(key=lambda item: item[0], reverse=True)
	return [candidate for _key, candidate in ranked[:max_jokers]]


def _build_covered_mask(h: int, w: int, selected: Sequence[JokerCandidate], sink) -> List[List[bool]]:
	covered = [[False] * w for _ in range(h)]
	sink.start(max(1, len(selected)), "marquage jokers")
	for candidate in selected:
		x1, y1, x2, y2 = candidate.bbox
		for y in range(y1, y2 + 1):
			row = covered[y]
			for x in range(x1, x2 + 1):
				row[x] = True
		sink.advance(1, desc="marquage jokers")
	return covered


def _append_completed_runs(active: dict[Tuple[int, int, int], List[int]], next_active: dict[Tuple[int, int, int], List[int]], rects: List[RectAction]) -> None:
	for key, run in active.items():
		if key not in next_active:
			x1, y1, x2, _prev_y, y2 = run
			rects.append((x1, y1, x2, y2, key[2]))


def _extend_runs_for_row(row: Sequence[int], covered_row: Sequence[bool], base_color: int, y: int, w: int) -> List[Tuple[int, int, int]]:
	runs: List[Tuple[int, int, int]] = []
	x = 0
	while x < w:
		if covered_row[x] or row[x] == base_color:
			x += 1
			continue

		color = row[x]
		x2 = x
		while x2 + 1 < w and not covered_row[x2 + 1] and row[x2 + 1] == color:
			x2 += 1
		runs.append((x, x2, color))
		x = x2 + 1

	return runs


def _build_residual_rectangles(target: Grid, base_color: int, covered: List[List[bool]], sink) -> List[RectAction]:
	h = len(target)
	w = len(target[0])
	active: dict[Tuple[int, int, int], List[int]] = {}
	rects: List[RectAction] = []

	sink.start(h, "construction rectangles")
	for y in range(h):
		sink.advance(1, desc="construction rectangles")
		row = target[y]
		runs = _extend_runs_for_row(row, covered[y], base_color, y, w)

		next_active: dict[Tuple[int, int, int], List[int]] = {}
		for x1, x2, color in runs:
			key = (x1, x2, color)
			if key in active and active[key][4] == y - 1:
				run = active[key]
				run[4] = y
				next_active[key] = run
			else:
				next_active[key] = [x1, y, x2, y, y]

		_append_completed_runs(active, next_active, rects)

		active = next_active

	for key, run in active.items():
		x1, y1, x2, _prev_y, y2 = run
		rects.append((x1, y1, x2, y2, key[2]))

	return rects


def _solution_strategy(dataset: dict, restart: int, iterations: int, seed: int, sink) -> Tuple[List[Action], dict]:
	target: Grid = dataset["grid"]
	h = len(target)
	w = len(target[0])
	max_jokers = dataset["maxJokers"]
	max_joker_size = dataset["maxJokerSize"]

	freq = color_frequencies(target)
	base_color = freq.most_common(1)[0][0]
	strategy = STRATEGIES[restart % len(STRATEGIES)]
	_ = random.Random(seed ^ (restart * 1_000_003) ^ (iterations * 97))

	candidates = _component_candidates(target, base_color, max_joker_size, sink)
	selected = _select_jokers(candidates, max_jokers=max_jokers, strategy=strategy, sink=sink)

	sink.set_total(h + len(candidates) + len(selected) + h + 1)
	covered = _build_covered_mask(h, w, selected, sink)
	residual_rects = _build_residual_rectangles(target, base_color, covered, sink)

	actions: List[Action] = [(0, 0, w - 1, h - 1, base_color)]
	actions.extend(residual_rects)
	actions.extend(("JOKER", x1, y1, x2, y2, -1) for x1, y1, x2, y2 in (candidate.bbox for candidate in selected))

	sink.advance(1, desc="validation")

	stats = {
		"strategy": strategy,
		"base_color": base_color,
		"candidates": len(candidates),
		"jokers": len(selected),
		"residual_rectangles": len(residual_rects),
		"rng_seed": seed,
	}
	return actions, stats


def _apply_progress_message(bars: dict[int, tqdm], msg: Tuple, show_progress: bool) -> None:
	kind = msg[0]
	if kind == "stop":
		return

	task_id = msg[1]
	bar = bars.get(task_id)

	if kind == "start":
		total, desc = msg[2], msg[3]
		if bar is None:
			bar = tqdm(total=total, desc=desc, position=task_id + 1, leave=False, dynamic_ncols=True, disable=not show_progress)
			bars[task_id] = bar
		else:
			bar.total = (bar.total or 0) + total
			bar.set_description_str(desc)
			bar.refresh()
	elif kind == "total" and bar is not None:
		bar.total = msg[2]
		bar.refresh()
	elif kind == "advance" and bar is not None:
		delta = msg[2]
		desc = msg[3]
		if desc:
			bar.set_description_str(desc)
		bar.update(delta)
	elif kind == "close" and bar is not None:
		desc = msg[2]
		if desc:
			bar.set_description_str(desc)
		bar.close()
		bars.pop(task_id, None)


def _monitor_progress(progress_queue, stop_event: threading.Event, show_progress: bool) -> None:
	bars: dict[int, tqdm] = {}
	while not stop_event.is_set():
		msg = progress_queue.get()
		if not msg:
			continue
		if msg[0] == "stop":
			break
		_apply_progress_message(bars, msg, show_progress)

	for bar in bars.values():
		bar.close()


def _build_sink(show_progress: bool, task_id: int, progress_queue=None, serial: bool = False):
	if not show_progress:
		return NullProgressSink()
	if progress_queue is not None:
		return QueueProgressSink(progress_queue, task_id=task_id, enabled=True)
	if serial:
		return LocalProgressSink(position=task_id + 1, desc=f"variant {task_id}", enabled=True)
	return NullProgressSink()


def _run_single_restart(dataset: dict, restart: int, iterations: int, seed: int, show_progress: bool, progress_queue=None) -> Tuple[List[Action], dict]:
	sink = _build_sink(show_progress, restart, progress_queue=progress_queue, serial=progress_queue is None)
	try:
		sink.start(0, f"variant {restart} [{STRATEGIES[restart % len(STRATEGIES)]}]")
		actions, stats = _solution_strategy(dataset, restart, iterations, seed, sink)
	finally:
		sink.close(f"variant {restart} terminé")
	return actions, stats


def _pick_better(candidate: List[Action], current_best: List[Action], dataset: dict) -> List[Action]:
	score_c, ok_c, _ = evaluate_actions(candidate, dataset)
	score_b, ok_b, _ = evaluate_actions(current_best, dataset)
	if not ok_c:
		return current_best
	if not ok_b:
		return candidate
	if score_c > score_b:
		return candidate
	if score_c == score_b and len(candidate) < len(current_best):
		return candidate
	return current_best


def find_best_solution(dataset: dict, iterations: int, restarts: int, seed: int, workers: int, show_progress: bool) -> Tuple[List[Action], dict]:
	cpu_count = os.cpu_count() or 1
	worker_count = min(max(1, workers), cpu_count, max(1, restarts))
	best_actions: List[Action] = []
	best_stats: dict = {}

	if worker_count <= 1 or restarts <= 1:
		return _find_best_solution_serial(dataset, iterations, restarts, seed, show_progress)

	return _find_best_solution_parallel(dataset, iterations, restarts, seed, worker_count, show_progress)


def _find_best_solution_serial(dataset: dict, iterations: int, restarts: int, seed: int, show_progress: bool) -> Tuple[List[Action], dict]:
	overall = tqdm(total=restarts, desc="dataset_5", position=0, dynamic_ncols=True, leave=True, disable=not show_progress)
	best_actions: List[Action] = []
	best_stats: dict = {}
	best_score = -1
	try:
		for restart in range(restarts):
			actions, stats = _run_single_restart(dataset, restart, iterations, seed, show_progress)
			score, ok, _ = evaluate_actions(actions, dataset)
			if ok and (score > best_score or (score == best_score and (not best_actions or len(actions) < len(best_actions)))):
				best_actions = actions
				best_stats = stats
				best_score = score
			overall.update(1)
	finally:
		overall.close()
	return best_actions, best_stats


def _find_best_solution_parallel(dataset: dict, iterations: int, restarts: int, seed: int, worker_count: int, show_progress: bool) -> Tuple[List[Action], dict]:
	manager = mp.Manager()
	progress_queue = manager.Queue() if show_progress else None
	stop_event = threading.Event()
	monitor = None
	if show_progress and progress_queue is not None:
		monitor = threading.Thread(target=_monitor_progress, args=(progress_queue, stop_event, show_progress), daemon=True)
		monitor.start()

	overall = tqdm(total=restarts, desc="dataset_5", position=0, dynamic_ncols=True, leave=True, disable=not show_progress)
	best_actions: List[Action] = []
	best_stats: dict = {}
	best_score = -1
	try:
		with ProcessPoolExecutor(max_workers=worker_count) as pool:
			futures = {
				pool.submit(_run_single_restart, dataset, restart, iterations, seed, show_progress, progress_queue): restart
				for restart in range(restarts)
			}

			for future in as_completed(futures):
				actions, stats = future.result()
				score, ok, _ = evaluate_actions(actions, dataset)
				if ok and (score > best_score or (score == best_score and (not best_actions or len(actions) < len(best_actions)))):
					best_actions = actions
					best_stats = stats
					best_score = score
				overall.update(1)
	finally:
		overall.close()
		if progress_queue is not None:
			progress_queue.put(("stop",))
		if monitor is not None:
			stop_event.set()
			monitor.join(timeout=5)

	return best_actions, best_stats


def run_solver(dataset_id: int, iterations: int, restarts: int, workers: int, write_solution: bool, show_progress: bool = True) -> List[Action]:
	if dataset_id != 5:
		raise ValueError("solver_dataset_5 ne supporte que --dataset 5")
	if iterations < 0:
		raise ValueError("--iterations doit etre >= 0")
	if restarts < 1:
		raise ValueError("--restarts doit etre >= 1")
	if workers < 1:
		raise ValueError("--workers doit etre >= 1")

	dataset_path = ROOT_DIR / "datasets" / "dataset_5.json"
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

	dataset = load_dataset(dataset_path)
	seed = random.SystemRandom().randint(1, 2_147_483_647)

	started = time.time()
	best_actions, stats = find_best_solution(dataset, iterations, restarts, seed, workers, show_progress)
	elapsed = time.time() - started

	solution_txt = actions_to_solution_text(best_actions)
	dataset_txt = json.dumps(dataset, ensure_ascii=False)
	score, is_valid, message = test_solution.get_solution_score(solution_txt, dataset_txt)

	print("---------------------------------")
	print("Dataset: dataset_5.json")
	print(f"Actions: {len(best_actions)}")
	print(f"Valid  : {is_valid}")
	print(f"Score  : {score:_}")
	print(f"Message: {message}")
	print(f"Temps  : {elapsed:.2f}s")
	print(f"Seed run: {seed}")
	print(f"CPU workers utilises: {min(max(1, workers), os.cpu_count() or 1, max(1, restarts))}")
	print(f"Stratégie gagnante: {stats.get('strategy', 'n/a')}")
	print(f"Jokers utilises: {stats.get('jokers', 0)}")
	print(f"Rectangles restants: {stats.get('residual_rectangles', 0)}")
	print("Approche: fond uniforme + composantes rentables couvertes par jokers + rectangles résiduels")
	print("---------------------------------")

	if write_solution:
		out = ROOT_DIR / "solutions" / "solution_5.txt"
		out.write_text(solution_txt + "\n", encoding="utf-8")
		print(f"Solution ecrite dans: {out}")

	return best_actions


def main() -> None:
	parser = argparse.ArgumentParser(description="Solver optimisé pour dataset_5 avec progression détaillée tqdm")
	parser.add_argument("--dataset", type=int, default=5, choices=[5], help="Dataset number (must be 5)")
	parser.add_argument("--iterations", type=int, default=5000, help="Paramètre de compatibilité / diversification")
	parser.add_argument("--restarts", type=int, default=8, help="Nombre de variantes évaluées")
	parser.add_argument("--workers", type=int, default=1, help="Nombre de processus CPU (1 = désactive le parallélisme)")
	parser.add_argument("--write", action="store_true", help="Write solution to solutions/solution_5.txt")
	progress_group = parser.add_mutually_exclusive_group()
	progress_group.add_argument("--progress", dest="progress", action="store_true", default=True, help="Affiche les barres tqdm détaillées")
	progress_group.add_argument("--no-progress", dest="progress", action="store_false", help="Désactive les barres tqdm")
	args = parser.parse_args()

	run_solver(
		dataset_id=args.dataset,
		iterations=args.iterations,
		restarts=args.restarts,
		workers=args.workers,
		write_solution=args.write,
		show_progress=args.progress,
	)


if __name__ == "__main__":
	main()




