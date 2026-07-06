# Rapport final V2 - Robustesse des GNN face aux perturbations du graphe

## 1. Résumé

Ce rapport étudie l'effet du choix de l'optimiseur sur un Graph Convolutional Network à deux couches entraîné sur Cora. Le protocole principal V2 est maintenant complet: 650 exécutions réelles, soit 10 graines, 5 optimiseurs, le graphe propre et 3 familles de perturbations à 5%, 10%, 20% et 30%. Les optimiseurs comparés sont Adam, AdamW, RMSProp, AdaGrad et SGD.

Sous ce protocole fixe, Adam obtient la meilleure moyenne globale de test accuracy sur les 130 lignes qui lui correspondent: 0.7898, contre 0.7881 pour RMSProp, 0.7635 pour AdamW, 0.7470 pour AdaGrad et 0.1404 pour SGD. La conclusion reste volontairement limitée: Adam est le choix le plus équilibré pour ce GCN, Cora, ces graines, ces hyperparamètres et ces perturbations aléatoires. Ce n'est pas une preuve qu'Adam est universellement meilleur pour tous les GNN.

## 2. Introduction et question de recherche

Les GNN combinent les attributs des noeuds et la structure du graphe. Sur Cora, chaque noeud représente un article scientifique, les arêtes représentent des citations et les attributs correspondent à un vecteur bag-of-words. Comme un GCN agrège l'information des voisins, le bruit dans les attributs ou dans les arêtes peut changer la convergence et la qualité finale de classification.

La question de recherche est:

**Pour un même GCN à deux couches sur Cora, comment le choix de l'optimiseur influence-t-il la précision, le macro F1, la convergence, le temps d'entraînement et la robustesse face à des perturbations aléatoires du graphe?**

## 3. Travaux liés

Kipf et Welling ont popularisé le GCN spectral simplifié pour la classification semi-supervisée sur graphes. Adam et AdamW sont des variantes adaptatives souvent utilisées dans les réseaux profonds, RMSProp adapte aussi le pas d'apprentissage à partir des gradients récents, AdaGrad accumule les gradients au cours du temps et SGD constitue une référence non adaptative. Les travaux sur la robustesse des GNN montrent que la structure du graphe et les attributs des noeuds peuvent influencer fortement la prédiction. Ici, les perturbations sont aléatoires et non adversariales.

## 4. Design expérimental

Le protocole principal exécuté est le protocole fixe Cora V2:

| Élément | Valeur |
|---|---|
| Dataset | Cora Citation Network |
| Modèle | GCN à deux couches |
| Graines | 42 à 51 |
| Optimiseurs | Adam, AdamW, RMSProp, AdaGrad, SGD |
| Époques | 200 |
| Hidden channels | 16 |
| Dropout | 0.5 |
| Learning rate | 0.01 |
| Weight decay | 0.0005 |
| Device | CPU local |
| Runs complétés | 650 / 650 |

Chaque ligne brute enregistre le commit Git, la graine de base, la graine résolue, le protocole, la perturbation, la sévérité demandée, la sévérité réellement appliquée, les métriques train/validation/test, le macro F1, la perte, le temps d'entraînement, l'époque du meilleur score validation et les métadonnées matériel.

## 5. Dataset et architecture GCN

Cora contient 2708 articles, 1433 attributs binaires de mots et 7 classes. Le split Planetoid standard est utilisé. Le modèle est un GCN à deux couches avec ReLU, dropout 0.5 et une couche finale de classification. Le même modèle et le même budget d'entraînement sont utilisés pour tous les optimiseurs afin de rendre la comparaison contrôlée.

## 6. Protocole fixe et protocole tuné

Le protocole fixe est celui réellement utilisé pour les résultats principaux. Il garde les mêmes hyperparamètres pour tous les optimiseurs. Cela favorise la comparabilité, pas forcément la performance maximale de chaque optimiseur.

Le dépôt contient aussi une configuration V2 tunée qui sélectionne les hyperparamètres avec la validation uniquement. Le test n'est pas utilisé pour le tuning. Cette extension est prête dans le pipeline, mais le rapport ne mélange pas ces résultats avec le protocole fixe.

## 7. Définitions des perturbations aléatoires

Trois perturbations V2 sont utilisées:

| Perturbation | Définition |
|---|---|
| Feature masking | Mise à zéro d'une fraction des entrées actives non nulles du vecteur d'attributs. |
| Edge removal | Suppression aléatoire d'une fraction des arêtes non orientées uniques, avec cohérence symétrique. |
| Fake edge addition | Ajout aléatoire d'arêtes entre paires de noeuds non connectées, sans boucle ni doublon. |

Les niveaux de sévérité sont 0.05, 0.10, 0.20 et 0.30. L'ancien bruit gaussien de V1 est conservé comme comportement legacy et n'est pas appelé pourcentage de corruption.

## 8. Corruption à l'entraînement et corruption à l'inférence

Le protocole principal exécuté ici est la corruption à l'entraînement: le graphe ou les attributs sont perturbés, puis le modèle est entraîné et évalué sur cette donnée perturbée. Son interprétation est la capacité à apprendre à partir d'un graphe bruité.

Le dépôt supporte aussi une étude de corruption à l'inférence: entraîner sur le graphe propre puis évaluer les poids fixes sous perturbation. Cette étude est séparée afin de ne pas mélanger deux questions scientifiques différentes.

## 9. Méthodologie statistique

Les agrégats V2 rapportent la moyenne, l'écart type, un intervalle de confiance 95% approximatif, le nombre de graines, la baisse par rapport au graphe propre et un score de robustesse AUC normalisé sur les niveaux de sévérité. Les classements sont descriptifs. Aucune phrase de significativité statistique n'est utilisée ici sans test apparié complet.

## 10. Résultats principaux

### 10.1 Graphe propre

| Optimiseur | Accuracy moyenne | IC95 accuracy | Macro F1 moyen | Graines |
|---|---:|---:|---:|---:|
| Adam | 0.8111 | ±0.0032 | 0.8004 | 10 |
| RMSProp | 0.8058 | ±0.0067 | 0.7964 | 10 |
| AdamW | 0.7871 | ±0.0052 | 0.7800 | 10 |
| AdaGrad | 0.7706 | ±0.0081 | 0.7640 | 10 |
| SGD | 0.1516 | ±0.0130 | 0.0757 | 10 |

Adam est premier sur le graphe propre, mais RMSProp reste proche. SGD échoue sous le learning rate fixe 0.01 et ne doit pas être présenté comme stable simplement parce que son gap train/validation est plus faible.

### 10.2 Moyenne globale sur les 650 runs

| Optimiseur | Accuracy moyenne | Macro F1 moyen | Temps moyen (s) | Meilleure époque validation |
|---|---:|---:|---:|---:|
| Adam | 0.7898 | 0.7791 | 7.76 | 141.1 |
| RMSProp | 0.7881 | 0.7777 | 10.63 | 107.5 |
| AdamW | 0.7635 | 0.7561 | 21.86 | 94.9 |
| AdaGrad | 0.7470 | 0.7410 | 12.03 | 151.4 |
| SGD | 0.1404 | 0.0668 | 8.37 | 67.2 |

La moyenne globale inclut le graphe propre et les graphes perturbés. Elle ne doit pas être comparée directement à un benchmark clean-only.

### 10.3 Robustesse par type de perturbation

| Perturbation | Adam Acc/F1 | RMSProp Acc/F1 | AdamW Acc/F1 | AdaGrad Acc/F1 | SGD Acc/F1 |
|---|---:|---:|---:|---:|---:|
| Feature masking | 0.7958 / 0.7847 | 0.7955 / 0.7851 | 0.7742 / 0.7674 | 0.7525 / 0.7461 | 0.1415 / 0.0624 |
| Edge removal | 0.7917 / 0.7807 | 0.7903 / 0.7792 | 0.7614 / 0.7544 | 0.7543 / 0.7467 | 0.1478 / 0.0761 |
| Fake edge addition | 0.7767 / 0.7665 | 0.7740 / 0.7642 | 0.7491 / 0.7406 | 0.7284 / 0.7245 | 0.1291 / 0.0597 |

Fake edge addition est la perturbation la plus pénalisante pour les optimiseurs adaptatifs, car elle ajoute des voisinages trompeurs. Edge removal est moins destructrice jusqu'à 30%, probablement grâce à la redondance locale du graphe Cora.

### 10.4 Détail par sévérité - feature masking

| Sévérité | Optimiseur | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 0.05 | Adam | 0.8039 | 0.7934 |
| 0.05 | RMSProp | 0.8004 | 0.7894 |
| 0.05 | AdamW | 0.7808 | 0.7747 |
| 0.05 | AdaGrad | 0.7756 | 0.7662 |
| 0.05 | SGD | 0.1356 | 0.0551 |
| 0.10 | Adam | 0.7998 | 0.7904 |
| 0.10 | RMSProp | 0.7968 | 0.7874 |
| 0.10 | AdamW | 0.7763 | 0.7703 |
| 0.10 | AdaGrad | 0.7616 | 0.7541 |
| 0.10 | SGD | 0.1650 | 0.0790 |
| 0.20 | RMSProp | 0.7974 | 0.7871 |
| 0.20 | Adam | 0.7953 | 0.7834 |
| 0.20 | AdamW | 0.7743 | 0.7668 |
| 0.20 | AdaGrad | 0.7423 | 0.7368 |
| 0.20 | SGD | 0.1352 | 0.0634 |
| 0.30 | RMSProp | 0.7875 | 0.7765 |
| 0.30 | Adam | 0.7840 | 0.7715 |
| 0.30 | AdamW | 0.7652 | 0.7578 |
| 0.30 | AdaGrad | 0.7303 | 0.7273 |
| 0.30 | SGD | 0.1301 | 0.0520 |

### 10.5 Détail par sévérité - edge removal

| Sévérité | Optimiseur | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 0.05 | Adam | 0.8057 | 0.7941 |
| 0.05 | RMSProp | 0.8016 | 0.7915 |
| 0.05 | AdamW | 0.7815 | 0.7755 |
| 0.05 | AdaGrad | 0.7616 | 0.7555 |
| 0.05 | SGD | 0.1581 | 0.0830 |
| 0.10 | Adam | 0.8019 | 0.7917 |
| 0.10 | RMSProp | 0.7936 | 0.7819 |
| 0.10 | AdamW | 0.7690 | 0.7625 |
| 0.10 | AdaGrad | 0.7673 | 0.7586 |
| 0.10 | SGD | 0.1274 | 0.0606 |
| 0.20 | RMSProp | 0.7892 | 0.7782 |
| 0.20 | Adam | 0.7852 | 0.7742 |
| 0.20 | AdamW | 0.7594 | 0.7506 |
| 0.20 | AdaGrad | 0.7429 | 0.7346 |
| 0.20 | SGD | 0.1413 | 0.0755 |
| 0.30 | RMSProp | 0.7767 | 0.7653 |
| 0.30 | Adam | 0.7740 | 0.7628 |
| 0.30 | AdaGrad | 0.7453 | 0.7382 |
| 0.30 | AdamW | 0.7358 | 0.7289 |
| 0.30 | SGD | 0.1643 | 0.0855 |

### 10.6 Détail par sévérité - fake edge addition

| Sévérité | Optimiseur | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 0.05 | Adam | 0.7999 | 0.7902 |
| 0.05 | RMSProp | 0.7974 | 0.7872 |
| 0.05 | AdamW | 0.7706 | 0.7633 |
| 0.05 | AdaGrad | 0.7554 | 0.7516 |
| 0.05 | SGD | 0.1236 | 0.0612 |
| 0.10 | Adam | 0.7878 | 0.7780 |
| 0.10 | RMSProp | 0.7834 | 0.7745 |
| 0.10 | AdamW | 0.7673 | 0.7581 |
| 0.10 | AdaGrad | 0.7505 | 0.7461 |
| 0.10 | SGD | 0.1413 | 0.0613 |
| 0.20 | RMSProp | 0.7696 | 0.7613 |
| 0.20 | Adam | 0.7667 | 0.7567 |
| 0.20 | AdamW | 0.7359 | 0.7276 |
| 0.20 | AdaGrad | 0.7210 | 0.7153 |
| 0.20 | SGD | 0.1293 | 0.0600 |
| 0.30 | Adam | 0.7523 | 0.7410 |
| 0.30 | RMSProp | 0.7454 | 0.7336 |
| 0.30 | AdamW | 0.7226 | 0.7134 |
| 0.30 | AdaGrad | 0.6868 | 0.6850 |
| 0.30 | SGD | 0.1222 | 0.0563 |

### 10.7 Gap train/validation, convergence et diagnostics

| Optimiseur | Train acc clean | Val acc clean | Gap | Loss clean |
|---|---:|---:|---:|---:|
| AdamW | 1.0000 | 0.7802 | 0.2198 | 0.0600 |
| RMSProp | 0.9986 | 0.7866 | 0.2120 | 0.3788 |
| Adam | 0.9993 | 0.7894 | 0.2099 | 0.3220 |
| AdaGrad | 0.9543 | 0.7524 | 0.2019 | 1.4247 |
| SGD | 0.1950 | 0.1438 | 0.0512 | 1.9442 |

| Optimiseur | Époque validation moyenne | Gradient L2 final moyen | Peak RSS CPU moyen (MB) | GPU peak (MB) |
|---|---:|---:|---:|---:|
| RMSProp | 112.9 | 0.2239 | 323.38 | 0.00 |
| Adam | 155.4 | 0.1426 | 322.97 | 0.00 |
| AdamW | 99.8 | 0.0406 | 323.24 | 0.00 |
| AdaGrad | 177.1 | 0.1183 | 323.78 | 0.00 |
| SGD | 70.4 | 0.0433 | 323.92 | 0.00 |

La mémoire GPU vaut 0 MB car ces expériences ont été exécutées sur CPU. La mesure mémoire utilisée dans le tableau est donc le peak RSS du processus Python pendant les diagnostics propres. Les courbes de gradients par époque sont disponibles dans `results/v2/diagnostics/v2_gradient_summary_clean.csv`.

## 11. Validation cross-dataset

Le dépôt contient la configuration cross-dataset pour Cora, CiteSeer et PubMed. Les résultats complets cross-dataset ne sont pas mélangés avec les résultats Cora principaux dans ce rapport. Les lignes absentes sont considérées comme du calcul restant à exécuter, pas comme des résultats estimés.

## 12. Discussion par optimiseur

Adam est le meilleur compromis observé dans le protocole fixe Cora: meilleure accuracy propre, meilleure moyenne globale et bonne robustesse face aux fausses arêtes.

RMSProp est très proche d'Adam. Il est parfois légèrement devant aux sévérités élevées en feature masking et edge removal, et atteint souvent son meilleur score validation plus tôt.

AdamW est performant sur le graphe propre mais reste derrière Adam et RMSProp dans la moyenne globale. Son loss clean faible et son gap train/validation élevé suggèrent un ajustement plus fort au petit ensemble d'entraînement.

AdaGrad est plus conservateur. Il reste exploitable, mais son plafond de performance est plus bas dans ce protocole fixe.

SGD sert de baseline non adaptative. Avec learning rate 0.01 et sans momentum dans le protocole fixe, il ne converge pas vers une bonne solution sur Cora.

## 13. Menaces à la validité et limites

Les perturbations sont aléatoires, pas adversariales. Les résultats dépendent de Cora, du split Planetoid, du GCN à deux couches, des hyperparamètres fixes, du budget de 200 époques et du matériel CPU local. Les temps d'entraînement ne doivent pas être généralisés à d'autres machines. Les intervalles de confiance sont descriptifs et ne remplacent pas une analyse statistique appariée complète.

## 14. Reproductibilité

Commandes principales:

```bash
make setup
make test
make lint
make format-check
make experiment-v2-cora
make aggregate-v2
make diagnostics-v2
make build-site
```

Les résultats principaux sont dans `results/v2/raw/v2_fixed_cora.csv` et `results/v2/aggregated/v2_aggregated_summary.csv`. Les diagnostics gradients/mémoire sont dans `results/v2/diagnostics/`. Le site statique copie ces fichiers dans `docs/assets/data/` et `docs/assets/downloads/`.

## 15. Références

- Kipf, T. N. and Welling, M. Semi-Supervised Classification with Graph Convolutional Networks.
- Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization.
- Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization.
- PyTorch Geometric documentation for Planetoid citation datasets.
- Cora citation network benchmark.

## 16. Annexe: fichiers et commandes

| Artefact | Chemin |
|---|---|
| Config principale | `configs/v2_fixed_cora.yaml` |
| Raw complet | `results/v2/raw/v2_fixed_cora.csv` |
| Agrégats | `results/v2/aggregated/v2_aggregated_summary.csv` |
| Diagnostics runs | `results/v2/diagnostics/v2_optimizer_diagnostics_runs_clean.csv` |
| Courbes gradients | `results/v2/diagnostics/v2_gradient_summary_clean.csv` |
| Rapport PDF | `reports/final_report_v2.pdf` |
| Notebook | `notebooks/GNN_Robustness_V2_Reproducibility.ipynb` |

Le rapport et le site ne doivent pas présenter V1 comme un protocole multi-graines. V1 reste une version legacy à une seule graine; V2 est le protocole reproductible utilisé pour les conclusions ci-dessus.
