# Robustesse des GNN sous perturbations aléatoires

Étude expérimentale de cinq optimiseurs sur un GCN à deux couches soumis à du bruit dans les attributs et la structure du graphe Cora.

Tableau de bord: <https://maaskk.github.io/gnn-optimization-under-graph-perturbations-cora/>

## Équipe

<table>
  <tr>
    <td align="center"><a href="https://github.com/mohamed-kar1"><img src="https://github.com/mohamed-kar1.png?size=96" width="72" alt="Mohamed Amine Kar-any"><br><sub><b>Mohamed Amine</b></sub></a></td>
    <td align="center"><a href="https://github.com/HamzaElhaddaji"><img src="https://github.com/HamzaElhaddaji.png?size=96" width="72" alt="Hamza Elhaddaji"><br><sub><b>Hamza</b></sub></a></td>
    <td align="center"><a href="https://github.com/Maaskk"><img src="https://github.com/Maaskk.png?size=96" width="72" alt="Ossama Ashad"><br><sub><b>Ossama</b></sub></a></td>
    <td align="center"><a href="https://github.com/Iliassouchida"><img src="https://github.com/Iliassouchida.png?size=96" width="72" alt="Iliass Ouchida"><br><sub><b>Iliass</b></sub></a></td>
    <td align="center"><a href="https://github.com/Mouhcine005"><img src="https://github.com/Mouhcine005.png?size=96" width="72" alt="Mouhcine Ayar"><br><sub><b>Mouhcine</b></sub></a></td>
  </tr>
</table>

Les contributions réalisées sont détaillées dans [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Protocole

- réseau: Cora
- modèle: GCN à deux couches
- optimiseurs: Adam, AdamW, RMSProp, AdaGrad et SGD
- graines: 42 à 51
- entraînement: 200 époques par run
- perturbations: masquage d'attributs, suppression d'arêtes et ajout de fausses arêtes
- niveaux de bruit: 5%, 10%, 20% et 30%

La matrice principale contient 5 optimiseurs, 13 conditions et 10 graines, soit 650 runs et 130 000 lignes d'historique d'entraînement.

Le projet étudie des perturbations aléatoires. Il ne simule pas une attaque optimisée contre le modèle.

## Reproduire l'étude

```bash
make setup
make test
make lint
make format-check
make smoke
make experiment-cora
make aggregate
make build-site
```

La génération de l'historique complet utilise:

```bash
.venv/bin/python scripts/collect_full_core_epoch_history.py --output-dir results/v2/proof
```

Le notebook [`notebooks/00_Preuve_Experimentale_GNN_Executee.ipynb`](notebooks/00_Preuve_Experimentale_GNN_Executee.ipynb) vérifie les runs, l'historique par époque, les agrégats et les tests complémentaires.

## Structure

| Chemin | Contenu |
| --- | --- |
| `configs/` | protocoles d'expérience |
| `src/gnn_robustness/` | données, modèle, perturbations, entraînement et statistiques |
| `scripts/` | exécution, diagnostics et génération des résultats |
| `results/` | sorties brutes et agrégées |
| `reports/` | rapport et figures |
| `deliverables/` | présentation et documents de soutenance |
| `docs/` | site statique |
| `tests/` | tests unitaires et contrats |

## Règles d'interprétation

- le jeu de test n'intervient pas dans le choix des hyperparamètres
- les comparaisons utilisent les graines appariées et leurs intervalles
- les faibles écarts numériques ne sont pas présentés comme des conclusions générales
- les temps d'exécution dépendent du matériel
- la visualisation interactive du graphe n'est pas une inférence GNN en direct

Le rapport final est disponible en [Markdown](reports/Final_Project_Report_GNN_Robustness.md) et en [PDF](reports/Final_Project_Report_GNN_Robustness.pdf).
