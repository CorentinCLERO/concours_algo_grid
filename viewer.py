import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

def apply_action(grid, action, target):
    """Applique une action RECT ou JOKER sur la grille."""
    parts = action.split()
    if parts[0] == "RECT":
        _, x1, y1, x2, y2, color = parts
        x1, y1, x2, y2, color = map(int, (x1, y1, x2, y2, color))
        grid[y1:y2+1, x1:x2+1] = color
    elif parts[0] == "JOKER":
        _, x1, y1, x2, y2 = parts
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        grid[y1:y2+1, x1:x2+1] = target[y1:y2+1, x1:x2+1]
    return grid

def compute_score(target, current, step, max_actions):
    """Calcule le score à une étape donnée."""
    total_pixels = target.size
    correct_pixels = np.sum(target == current)
    
    if correct_pixels < total_pixels:
        return round(1_000_000 * correct_pixels / total_pixels)
    else:
        return round(1_000_000 * max_actions / step)

def viewer(input_file, solution_file):
    # Charger dataset
    with open(input_file, "r") as f:
        data = json.load(f)
    target = np.array(data["grid"])
    max_actions = data["maxActions"]

    # Charger solution
    with open(solution_file, "r") as f:
        actions = [line.strip() for line in f if line.strip()]

    # Préparer états successifs
    states = [np.zeros_like(target)]
    for action in actions:
        new_state = states[-1].copy()
        new_state = apply_action(new_state, action, target)
        states.append(new_state)

    # Création figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)  # pour laisser de la place au slider

    pixel_colormap = matplotlib.colors.ListedColormap(['#000000', '#FFFFFF', '#1E93FF', '#F93C31', '#4FCC30', '#FFDC00', '#E53AA3', '#FF851B'])

    # Images initiales
    im_target = axes[0].imshow(target, cmap=pixel_colormap, vmin=0, vmax=7)
    axes[0].set_title("Fresque cible")

    im_solution = axes[1].imshow(states[0], cmap=pixel_colormap, vmin=0, vmax=7)
    axes[1].set_title("Solution - étape 0")

    for ax in axes:
            # Cadres noirs autour des images
            ax.axis("on")
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])

    # Titre global (score affiché ici)
    title = fig.suptitle(f"Score étape 0 : {compute_score(target, states[0], 1, max_actions):.3f}", fontsize=14)

    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.05])
    slider = Slider(ax_slider, "Étape", 0, len(actions), valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
        im_solution.set_data(states[step])
        axes[1].set_title(f"Solution - étape {step}")
        score = compute_score(target, states[step], max(1, step), max_actions)
        title.set_text(f"Score étape {step} : {score:.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python viewer.py <n>")
        print("Exemple: python viewer.py 1")
        sys.exit(1)

    try:
        dataset_id = int(sys.argv[1])
        if dataset_id <= 0:
            raise ValueError
    except ValueError:
        print("Erreur: <n> doit etre un entier positif (ex: 1, 2, 3).")
        sys.exit(1)

    root = Path(__file__).resolve().parent
    dataset_path = root / "datasets" / f"dataset_{dataset_id}.json"
    solution_path = root / "solutions" / f"solution_{dataset_id}.txt"

    if not dataset_path.exists():
        print(f"Erreur: dataset introuvable: {dataset_path}")
        sys.exit(1)
    if not solution_path.exists():
        print(f"Erreur: solution introuvable: {solution_path}")
        sys.exit(1)

    viewer(str(dataset_path), str(solution_path))


if __name__ == "__main__":
    main()
