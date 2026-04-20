# Botksy

Dans un futur proche, toute forme de street art a été **interdite**. Pourtant, des artistes rebelles refusent de disparaître et ont mis au point une flotte de drones clandestins capables de reproduire des fresques pixélisées sur les murs de la ville.  

Vous êtes chargé·e de programmer ces drones afin de **reproduire fidèlement des dessins interdits** dans un environnement où chaque action compte : les autorités surveillent et vos ressources sont limitées.  

Chaque jeu de données vous rapportera jusqu’à 1 million de points pour le classement (voir détail plus bas).

---

## Entrée

Chaque fresque à dessiner est fournie sous la forme d’une grille de pixels, avec des couleurs numérotées de `0` à `7`.
Votre programme devra soumettre une séquence d’actions permettant de reproduire cette fresque à l’identique, en partant d'une fresque vierge de même taille où toutes les cases valent initialement `0`.

Le format JSON des fichiers fournis est détaillé ci-dessous :  

```json
{
    "comment": "Exemple de fresque",   // Description courte du dataset
    "maxActions": 8,                   // Nombre maximal d'actions autorisées (RECT + JOKER)
    "maxJokers": 2,                    // Nombre maximal d'actions JOKER
    "maxJokerSize": 8,                 // Surface maximale de chaque JOKER
    "grid": [
        [1, 1, 1, 1, 1, 2, 0, 2, 1],   // Ligne y=0
        [1, 2, 0, 3, 1, 0, 1, 0, 1],   // Ligne y=1
        [1, 0, 1, 0, 1, 3, 0, 3, 1],   // Ligne y=2
        [1, 3, 0, 2, 1, 1, 1, 1, 1]    // Ligne y=3
    ]
}
```

La grille `grid` contient la couleur attendue pour chaque pixel du dessin (valeur entière entre 0 et 7).
L’indice `y=0` correspond à la première ligne et `x=0` à la première colonne. Le point `0 0` se trouve donc **en haut à gauche** de la grille.

---

## Sortie

Pour chaque fresque, vous devez soumettre votre solution sous la forme d’un fichier texte (`.txt`) contenant **une action par ligne**.  

Deux types d’actions sont disponibles :  

- `RECT <x1> <y1> <x2> <y2> <color>`  
  Colore tous les pixels du rectangle défini par les coins `(x1, y1)` et `(x2, y2)` avec la couleur `color`.

- `JOKER <x1> <y1> <x2> <y2>`  
  Permet de remplir instantanément une zone rectangulaire aux couleurs exactes attendues. L’usage est limité en nombre (`maxJokers`) et en surface (`maxJokerSize`). Chaque JOKER peut utiliser des tailles différentes, du moment que la surface maximum n'est pas dépassée. Dans l'exemple ci-dessus, `maxJokerSize` vaut 100 : une action `JOKER` pourrait donc peindre une zone de taille 10x10 ou 12x4, mais pas 8x14.

Dans les deux types d'action, les coordonnées des coins sont **incluses** dans le rectangle : une action `JOKER 0 0 2 2` affectera par exemple un carré de taille 3x3.

Exemple de sortie (`solution.txt`) :  

```
RECT 0 0 8 3 1
JOKER 5 0 7 1
JOKER 1 1 3 2
RECT 5 2 5 2 3
RECT 6 2 6 2 0
RECT 7 2 7 2 3
RECT 1 3 1 3 3
RECT 2 3 2 3 0
```

---

## Calcul du score

À la fin de la liste d’actions, on évalue le dessin produit par vos drones par rapport à l’image cible.

- Si l’image n’est pas parfaitement reproduite :  
  **score = nbPixelsCorrects / nbPixelsTotaux × 1.000.000**  
  (score ≤ 1.000.000).

- Si l’image est parfaitement reproduite :  
  **score = NbActionsAutorisées / NbActionsUtilisées × 1.000.000**
  (score ≥ 1.000.000).

Votre objectif est donc en priorité de reproduire fidèlement la fresque attendue. Pour certains jeux de données (mais probablement pas tous !) vous arriverez à reproduire exactement la fresque : pour marquer plus de points, il faudra optimiser le nombre d'actions.

L'exemple ci-dessus parvient à reconstituer 35 pixels sur les 36 de la fresque originale, cette solution marquerait donc 972.222 points.

---

## Classement global

Pour chaque jeu de données, le joueur avec la meilleure solution remporte 1 million de points de classement. Le score des autres joueurs est calculé en fonction de la solution du meilleur joueur avec la formule suivante :

```
1.000.000 * score_joueur / score_meilleur
```

Par exemple, si votre solution est deux fois moins efficace que celle du meilleur adversaire, votre score sur ce jeu de données sera de 500 000 points.  

Le classement est déterminé en fonction du total de points obtenus. Seule votre meilleure soumission sur chaque jeu de données sera prise en compte. Vous pouvez soumettre autant de solutions que vous souhaitez.
