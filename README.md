# concours_algo_grid

## Format des datasets / solutions

- dataset_X.json
- solution_X.txt

X est un entier de 1 à 6

## Tester une solution 

```bash
python testsolution.py X
```

## Viewer

```bash
python viewer.py X
```

## Solvers (en greedy + monte carlo)

```bash
python run-worker --dataset X --iterations 5000 --restarts 4 --workers 4 --write
```

Remplacer X par le numéro du dataset à résoudre (1 à 6).

Le script écrit la meilleure solution trouvée dans `solutions/solution_X.txt` si le param `--write` est passé.

`--workers` permet d'utiliser plusieurs processus CPU (mettre `1` pour désactiver le parallélisme, éviter de mettre le max de processeurs logiques pour éviter que windows explose).
