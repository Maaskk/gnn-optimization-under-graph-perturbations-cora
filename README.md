# Robustesse des Réseaux de Neurones de Graphes sous Perturbations Aléatoires

Ce dépôt contient le projet académique final du Projet 13, Option 4:
**Robustesse des GNN face au bruit dans le graphe**.

Tableau de bord: <https://maaskk.github.io/gnn-optimization-under-graph-perturbations-cora/>

## Question de recherche

Sur le réseau de citations Cora, comment Adam, AdamW, RMSProp, AdaGrad et SGD se comparent-ils avec la même architecture GCN à deux couches lorsque des perturbations aléatoires sont appliquées aux attributs ou à la structure du graphe?

L'étude porte uniquement sur des perturbations aléatoires. Elle ne met pas en place d'attaques optimisées contre le modèle.

## Étude finale

- Dataset: Cora citation network.
- Modèle: Graph Convolutional Network à deux couches.
- Optimiseurs: Adam, AdamW, RMSProp, AdaGrad, SGD.
- Graines principales: 42 à 51.
- Budget d'entraînement: 200 époques par run.
- Conditions principales par optimiseur: graphe propre, masquage d'attributs à 5%, 10%, 20%, 30%, suppression d'arêtes à 5%, 10%, 20%, 30%, ajout de fausses arêtes à 5%, 10%, 20%, 30%.
- Matrice principale: 5 optimiseurs x 13 conditions x 10 graines = 650 runs réels.

Des validations complémentaires sont générées séparément pour les tests multi-datasets, le tuning basé uniquement sur la validation et la robustesse à l'inférence.

## Perturbations

- `feature_masking`: met à zéro une fraction demandée des entrées actives non nulles.
- `edge_removal`: supprime une fraction demandée des connexions non orientées uniques et reconstruit une représentation symétrique.
- `fake_edge_addition`: insère des paires de noeuds précédemment non connectées, sans self-loops ni doublons, et reconstruit une représentation symétrique.

## Reproductibilité

```bash
make setup
make test
make lint
make format-check
make smoke
make experiment-cora
make experiment-cross-dataset
make experiment-tuned
make experiment-inference
make aggregate
make build-site
make reproduce-final
```

La commande smoke écrit uniquement dans `results/ci_smoke/` et n'écrase pas les résultats finaux.

## Structure du dépôt

| Chemin | Rôle |
| --- | --- |
| `configs/` | Configurations des protocoles fixe, tuné, multi-datasets et inférence |
| `src/gnn_robustness/` | Chargement des données, modèle GCN, perturbations, entraînement, agrégation, statistiques, métadonnées |
| `scripts/` | Runners d'expériences, diagnostics, statistiques, génération du rapport et données du site |
| `results/v2/raw/` | Lignes brutes générées par les expériences |
| `results/v2/aggregated/` | Table brute combinée et résumés agrégés |
| `results/v2/diagnostics/` | Diagnostics de gradients, temps et mémoire |
| `reports/` | Rapport final, sources PDF et visualisations générées |
| `deliverables/` | Présentation, script de soutenance, Q&A et QR codes |
| `docs/` | Site statique GitHub Pages |
| `notebooks/` | Notebook de reproductibilité |
| `tests/` | Tests unitaires et contrats |

## Garde-fous scientifiques

- Le split test n'est pas utilisé pour choisir les hyperparamètres.
- Les différences sont interprétées avec les intervalles et les comparaisons appariées.
- Les petits écarts numériques restent prudents.
- SGD est discuté uniquement dans le protocole fixe choisi, pas comme conclusion générale.
- Les temps dépendent du matériel local.
- L'animation du graphe dans le site est une visualisation interactive illustrative, pas une inférence GNN en direct.

## Artefacts principaux

- `reports/Final_Project_Report_GNN_Robustness.md`
- `reports/Final_Project_Report_GNN_Robustness.tex`
- `reports/Final_Project_Report_GNN_Robustness.pdf`
- `deliverables/GCN_Robustness_Final_Defense.pptx`
- `deliverables/GCN_Robustness_Final_Defense.pdf`
- `deliverables/Script_Soutenance_GNN_FR.md`
- `deliverables/Questions_Reponses_Jury_GNN_FR.md`
