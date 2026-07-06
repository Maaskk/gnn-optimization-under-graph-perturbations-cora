# Guide de reproductibilité

## Environnement

Utiliser Python 3.11 ou 3.12. Installer les dépendances avec:

```bash
make setup
```

Les runners d'expériences écrivent les métadonnées matérielles et logicielles dans le dossier de métadonnées du pipeline. Chaque ligne brute contient l'identifiant de version, la graine résolue, le taux de perturbation demandé, le taux réellement appliqué et le chemin des métadonnées matérielles.

## Vérifications qualité

```bash
make test
make lint
make format-check
```

## Smoke test isolé

```bash
make smoke
```

La commande smoke écrit uniquement dans `results/ci_smoke/`. Elle valide le pipeline sans remplacer les matrices scientifiques finales.

## Matrice principale Cora

```bash
make experiment-cora
make aggregate
make diagnostics
make build-site
```

La matrice principale Cora contient 650 lignes générées:

- Cora
- 10 graines
- 5 optimiseurs
- graphe propre, feature masking, suppression d'arêtes et ajout de fausses arêtes
- sévérités 0.05, 0.10, 0.20, 0.30
- 200 époques par run

## Validations complémentaires

```bash
make experiment-cross-dataset
make experiment-tuned
make experiment-inference
make aggregate
```

La validation multi-datasets exécute Cora, CiteSeer et PubMed avec cinq graines et quatre conditions. Le protocole tuné utilise uniquement les métriques de validation, puis verrouille les hyperparamètres avant l'évaluation test finale. La robustesse à l'inférence entraîne sur Cora propre puis perturbe uniquement l'évaluation.

## Diagnostics optimiseurs

```bash
make diagnostics
```

Cette commande entraîne chaque optimiseur sur Cora propre, journalise la norme L2 du gradient à chaque époque et enregistre les résumés mémoire CPU/GPU.

## Reproduction finale

```bash
make reproduce-final
```

Cette commande est volontairement longue: elle lance les tests, le lint, la vérification de format, les protocoles expérimentaux finaux, l'agrégation, les diagnostics et la génération du site.

Ne jamais compléter les lignes manquantes manuellement. Seules les lignes brutes générées par le pipeline doivent être agrégées ou déclarées terminées.
