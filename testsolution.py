import sys
from pathlib import Path
import test_solution


def main():
    if len(sys.argv) != 2:
        print("Usage: python testsolution.py <numero_dataset>")
        print("Exemple: python testsolution.py 1")
        sys.exit(1)

    dataset_id = sys.argv[1]

    root = Path(__file__).resolve().parent
    solution_file = root / "solutions" / f"solution_{dataset_id}.txt"
    dataset_file = root / "datasets" / f"dataset_{dataset_id}.json"

    if not solution_file.exists():
        print(f"❌ Fichier solution introuvable: {solution_file}")
        sys.exit(1)

    if not dataset_file.exists():
        print(f"❌ Fichier dataset introuvable: {dataset_file}")
        sys.exit(1)

    solution = solution_file.read_text(encoding="utf-8")
    dataset = dataset_file.read_text(encoding="utf-8")

    score, is_valid, message = test_solution.get_solution_score(solution, dataset)

    print("---------------------------------")
    print(f"Solution: {solution_file.name}")
    print(f"Dataset : {dataset_file.name}")
    print("---------------------------------")

    if is_valid:
        print("✅ Solution valide")
        print(f"Score: {score:_}")
    else:
        print("❌ Solution invalide")

    print(f"Message: {message}")


if __name__ == "__main__":
    main()
