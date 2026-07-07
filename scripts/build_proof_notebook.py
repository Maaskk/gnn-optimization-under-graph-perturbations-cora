from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "00_Preuve_Experimentale_GNN_Executee.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook["cells"] = [
        md(
            """
            # Preuve expérimentale exécutée - GNN Cora

            Ce notebook est l'artefact de défense du projet. Il montre le chemin complet:
            données Cora -> modèle GCN -> perturbations -> 650 runs Cora -> 130000 lignes
            d'historique epoch-par-epoch -> agrégats -> limites.

            Le notebook est en français et doit être exécuté de haut en bas. Les cellules
            lèvent une erreur si les fichiers de preuve ne contiennent pas la grille attendue.
            """
        ),
        md(
            """
            ## 1. Objectif du projet

            Le projet compare cinq optimiseurs (`Adam`, `AdamW`, `RMSProp`, `AdaGrad`, `SGD`)
            pour entraîner le même GCN à deux couches sur le réseau de citations Cora.

            La question étudiée est: sous un protocole fixe, quel optimiseur reste le plus
            stable quand on applique des perturbations aléatoires au graphe ou aux attributs?

            Les perturbations sont non adversariales:
            - masquage aléatoire de caractéristiques actives non nulles;
            - suppression aléatoire d'arêtes existantes;
            - ajout aléatoire de fausses arêtes valides.

            Ce projet ne prétend pas mesurer une robustesse adversariale universelle.
            """
        ),
        md("## 2. Environnement et reproductibilité"),
        code(
            """
            from __future__ import annotations

            import inspect
            import json
            import math
            import os
            import platform
            import subprocess
            import sys
            import time
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import torch
            import torch.nn.functional as F
            import torch_geometric
            from IPython.display import Markdown, display

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = Path.cwd().parents[0]
            os.chdir(ROOT)
            SRC = ROOT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            from gnn_robustness.data import cora_summary, load_planetoid
            from gnn_robustness.metrics import accuracy_score, macro_f1_score
            from gnn_robustness.model import GCN
            from gnn_robustness.optimizers import DEFAULT_OPTIMIZERS, make_optimizer
            from gnn_robustness.train import set_seed
            from gnn_robustness.v2_config import load_v2_config, resolved_seed
            from gnn_robustness.v2_perturbations import (
                add_undirected_fake_edges,
                mask_active_features,
                remove_undirected_edges,
                unique_undirected_edges,
            )
            from gnn_robustness.v2_results import aggregate_raw_results

            def git_output(*args: str) -> str:
                try:
                    return subprocess.check_output(["git", *args], text=True).strip()
                except Exception:
                    return "indisponible"

            config = load_v2_config("configs/v2_fixed_cora.yaml")
            env_rows = [
                ("Python", sys.version.split()[0]),
                ("PyTorch", torch.__version__),
                ("PyTorch Geometric", torch_geometric.__version__),
                ("CUDA disponible", str(torch.cuda.is_available())),
                ("Device protocole", config.training.device),
                ("OS", platform.platform()),
                ("CPU", platform.processor() or platform.machine()),
                ("Git commit", git_output("rev-parse", "--short", "HEAD")),
                ("Graines", ", ".join(map(str, config.seeds))),
                ("Commande historique complet", "python scripts/collect_full_core_epoch_history.py --output-dir results/v2/proof"),
                ("Commande notebook", "python -m jupyter nbconvert --execute --to notebook --inplace notebooks/00_Preuve_Experimentale_GNN_Executee.ipynb"),
            ]
            display(pd.DataFrame(env_rows, columns=["Élément", "Valeur"]))
            """
        ),
        md(
            """
            ## 3. Comprendre les données Cora

            Dans Cora:
            - un noeud représente un article scientifique;
            - une arête représente une relation de citation utilisée par le passage de messages;
            - une caractéristique est un attribut lié aux mots du document;
            - une classe est une catégorie de recherche;
            - les masques `train`, `validation`, `test` définissent les noeuds utilisés pour apprendre,
              choisir le meilleur comportement et évaluer.
            """
        ),
        code(
            """
            dataset, data = load_planetoid("Cora", "data")
            summary = cora_summary(dataset, data)
            unique_edges = len(unique_undirected_edges(data.edge_index))
            summary_rows = [
                ("Noeuds", summary["num_nodes"]),
                ("Arêtes orientées dans edge_index", int(data.edge_index.size(1))),
                ("Arêtes non orientées uniques", unique_edges),
                ("Shape edge_index", tuple(data.edge_index.shape)),
                ("Shape features x", tuple(data.x.shape)),
                ("Nombre de classes", summary["num_classes"]),
                ("Noeuds train", summary["train_nodes"]),
                ("Noeuds validation", summary["validation_nodes"]),
                ("Noeuds test", summary["test_nodes"]),
            ]
            display(pd.DataFrame(summary_rows, columns=["Propriété", "Valeur"]))

            degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            sample_nodes = pd.DataFrame(
                {
                    "node_id": list(range(12)),
                    "label_classe": data.y[:12].tolist(),
                    "degré_sortant_edge_index": degrees[:12].tolist(),
                    "nb_features_actives": (data.x[:12] != 0).sum(dim=1).tolist(),
                    "split": [
                        "train" if bool(data.train_mask[i]) else "validation" if bool(data.val_mask[i]) else "test" if bool(data.test_mask[i]) else "hors split"
                        for i in range(12)
                    ],
                }
            )
            display(sample_nodes)
            """
        ),
        md("## 4. Comprendre le GCN"),
        code(
            """
            print(inspect.getsource(GCN))
            model = GCN(
                input_channels=dataset.num_node_features,
                hidden_channels=config.training.hidden_channels,
                output_channels=dataset.num_classes,
                dropout=config.training.dropout,
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            display(pd.DataFrame(
                [
                    ("Entrée", dataset.num_node_features),
                    ("Couche cachée", config.training.hidden_channels),
                    ("Sortie", dataset.num_classes),
                    ("Dropout", config.training.dropout),
                    ("Paramètres entraînables", param_count),
                    ("Loss réelle", "torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])"),
                ],
                columns=["Élément", "Valeur"],
            ))
            display(Markdown(
                "Le modèle utilise deux couches GCN: la première apprend une représentation locale "
                "à partir des voisins, la seconde transforme cette représentation en scores de classes."
            ))
            """
        ),
        md("## 5. Comprendre les optimiseurs"),
        code(
            """
            optimizer_rows = []
            purposes = {
                "Adam": "Adaptatif; combine moyenne du gradient et variance. Souvent très stable.",
                "AdamW": "Variante d'Adam avec découplage du weight decay.",
                "RMSProp": "Adaptatif; normalise les gradients avec une moyenne glissante.",
                "AdaGrad": "Adapte le pas selon l'historique; peut ralentir fortement.",
                "SGD": "Descente de gradient simple; protocole fixe potentiellement défavorable sans tuning.",
            }
            for name in config.optimizers:
                optimizer_rows.append(
                    {
                        "optimiseur": name,
                        "learning_rate": config.training.learning_rate,
                        "weight_decay": config.training.weight_decay,
                        "momentum": 0.0,
                        "explication orale": purposes[name],
                    }
                )
            display(pd.DataFrame(optimizer_rows))
            print(inspect.getsource(make_optimizer))
            """
        ),
        md("## 6. Comprendre les perturbations"),
        code(
            """
            print(inspect.getsource(mask_active_features))
            print(inspect.getsource(remove_undirected_edges))
            print(inspect.getsource(add_undirected_fake_edges))
            """
        ),
        code(
            """
            severity = 0.20
            seed = 123

            feature_result = mask_active_features(data.x, severity=severity, seed=seed)
            active_before = int((data.x != 0).sum().item())
            active_after = int((feature_result.features != 0).sum().item())

            removal_result = remove_undirected_edges(data.edge_index, severity=severity, seed=seed)
            before_pairs = unique_undirected_edges(data.edge_index)
            removed_pairs = unique_undirected_edges(removal_result.edge_index)

            addition_result = add_undirected_fake_edges(
                data.edge_index,
                num_nodes=int(data.num_nodes),
                severity=severity,
                seed=seed,
            )
            added_pairs = unique_undirected_edges(addition_result.edge_index)
            inserted_pairs = added_pairs - before_pairs

            assert active_after == active_before - feature_result.metadata["masked_feature_entries"]
            assert removed_pairs.issubset(before_pairs)
            assert len(before_pairs - removed_pairs) == removal_result.metadata["actual_removed_edges"]
            assert not any(source == target for source, target in inserted_pairs)
            assert inserted_pairs.isdisjoint(before_pairs)
            assert len(inserted_pairs) == addition_result.metadata["actual_inserted_edges"]

            perturbation_checks = pd.DataFrame(
                [
                    {
                        "perturbation": "feature_masking",
                        "avant": active_before,
                        "après": active_after,
                        "preuve": "les entrées actives non nulles masquées deviennent zéro",
                    },
                    {
                        "perturbation": "edge_removal",
                        "avant": len(before_pairs),
                        "après": len(removed_pairs),
                        "preuve": "seules des arêtes existantes sont supprimées",
                    },
                    {
                        "perturbation": "fake_edge_addition",
                        "avant": len(before_pairs),
                        "après": len(added_pairs),
                        "preuve": "pas de self-loop, pas de doublon, pas d'arête déjà présente",
                    },
                ]
            )
            display(perturbation_checks)
            """
        ),
        code(
            """
            toy_edges = torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
                dtype=torch.long,
            )
            toy_removed = remove_undirected_edges(toy_edges, severity=0.25, seed=7).edge_index
            toy_added = add_undirected_fake_edges(toy_edges, num_nodes=5, severity=0.25, seed=7).edge_index
            positions = {
                0: (0.0, 0.0),
                1: (1.0, 0.8),
                2: (2.0, 0.0),
                3: (3.0, 0.8),
                4: (4.0, 0.0),
            }

            def draw_graph(ax, edge_index, title):
                pairs = sorted(unique_undirected_edges(edge_index))
                for source, target in pairs:
                    xs = [positions[source][0], positions[target][0]]
                    ys = [positions[source][1], positions[target][1]]
                    ax.plot(xs, ys, color="#9aa7b2", linewidth=2)
                for node, (x_pos, y_pos) in positions.items():
                    ax.scatter([x_pos], [y_pos], s=260, color="#a51f36", zorder=3)
                    ax.text(x_pos, y_pos, str(node), color="white", ha="center", va="center", weight="bold")
                ax.set_title(title)
                ax.axis("off")

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            draw_graph(axes[0], toy_edges, "Avant")
            draw_graph(axes[1], toy_removed, "Après suppression")
            draw_graph(axes[2], toy_added, "Après ajout de fausses arêtes")
            plt.show()
            """
        ),
        md(
            """
            ## 7. Preuve des 650 runs et des 130000 epochs

            Cette section charge l'historique complet produit par:

            `python scripts/collect_full_core_epoch_history.py --output-dir results/v2/proof`

            Elle vérifie:
            - 5 optimiseurs;
            - 10 graines;
            - 13 conditions Cora: clean + 3 perturbations x 4 sévérités;
            - 650 runs;
            - 200 epochs par run;
            - 130000 lignes epoch-par-epoch.
            """
        ),
        code(
            """
            proof_epoch_path = ROOT / "results/v2/proof/full_core_epoch_history.csv"
            proof_run_path = ROOT / "results/v2/proof/full_core_run_results.csv"
            proof_manifest_path = ROOT / "results/v2/proof/full_core_epoch_history_manifest.json"

            assert proof_epoch_path.exists(), f"Fichier manquant: {proof_epoch_path}"
            assert proof_run_path.exists(), f"Fichier manquant: {proof_run_path}"
            assert proof_manifest_path.exists(), f"Fichier manquant: {proof_manifest_path}"

            epoch_history = pd.read_csv(proof_epoch_path)
            proof_runs = pd.read_csv(proof_run_path)
            proof_manifest = json.loads(proof_manifest_path.read_text(encoding="utf-8"))

            expected_runs = len(config.optimizers) * len(config.seeds) * (1 + 3 * len(config.severities))
            expected_epochs = expected_runs * config.training.epochs
            assert expected_runs == 650
            assert expected_epochs == 130000
            assert len(proof_runs) == expected_runs, len(proof_runs)
            assert len(epoch_history) == expected_epochs, len(epoch_history)

            per_run_epochs = epoch_history.groupby(
                ["optimizer", "seed", "perturbation_type", "requested_severity"],
                dropna=False,
            )["epoch"].nunique()
            assert int(per_run_epochs.min()) == config.training.epochs
            assert int(per_run_epochs.max()) == config.training.epochs

            display(pd.DataFrame(
                [
                    ("runs attendus", expected_runs),
                    ("runs observés", len(proof_runs)),
                    ("epochs par run", config.training.epochs),
                    ("lignes epoch attendues", expected_epochs),
                    ("lignes epoch observées", len(epoch_history)),
                    ("min epochs/run", int(per_run_epochs.min())),
                    ("max epochs/run", int(per_run_epochs.max())),
                    ("commit génération", proof_manifest.get("git_commit")),
                ],
                columns=["Contrôle", "Valeur"],
            ))
            display(epoch_history.head(12))
            """
        ),
        code(
            """
            selected = epoch_history[
                (epoch_history["optimizer"] == "Adam")
                & (epoch_history["seed"] == 42)
                & (epoch_history["perturbation_type"] == "clean")
                & (epoch_history["requested_severity"] == 0.0)
            ].sort_values("epoch")
            assert len(selected) == 200

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(selected["epoch"], selected["train_loss"], color="#a51f36")
            axes[0].set_title("Adam clean seed 42 - loss train")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("loss")

            axes[1].plot(selected["epoch"], selected["validation_loss"], color="#1f7a7a")
            axes[1].set_title("Adam clean seed 42 - loss validation")
            axes[1].set_xlabel("epoch")

            axes[2].plot(selected["epoch"], selected["validation_accuracy"], color="#13213c")
            axes[2].set_title("Adam clean seed 42 - accuracy validation")
            axes[2].set_xlabel("epoch")
            axes[2].set_ylim(0, 1)
            plt.tight_layout()
            plt.show()
            display(Markdown(
                "Ces courbes proviennent du fichier complet epoch-par-epoch, pas d'une image dessinée à la main."
            ))
            """
        ),
        md("## 8. Vérification des résultats principaux"),
        code(
            """
            raw_core = pd.read_csv(ROOT / "results/v2/raw/v2_fixed_cora.csv")
            display(raw_core.head(10)[
                ["seed", "optimizer", "perturbation_type", "requested_severity", "test_accuracy", "macro_f1", "final_epoch"]
            ])

            expected_optimizers = set(config.optimizers)
            expected_seeds = set(config.seeds)
            expected_conditions = {("clean", 0.0)}
            for perturbation in ["feature_masking", "edge_removal", "fake_edge_addition"]:
                for severity_value in config.severities:
                    expected_conditions.add((perturbation, float(severity_value)))

            observed_conditions = {
                (row.perturbation_type, round(float(row.requested_severity), 8))
                for row in raw_core.itertuples()
            }
            assert set(raw_core["optimizer"]) == expected_optimizers
            assert set(raw_core["seed"]) == expected_seeds
            assert observed_conditions == {
                (name, round(value, 8)) for name, value in expected_conditions
            }
            assert len(raw_core) == 650
            assert (raw_core["final_epoch"] == 200).all()

            grid_counts = raw_core.groupby(["perturbation_type", "requested_severity"]).size().reset_index(name="runs")
            display(grid_counts)
            display(pd.DataFrame(
                [
                    ("formule", "5 optimiseurs x 13 conditions x 10 graines"),
                    ("calcul", 5 * 13 * 10),
                    ("lignes brutes observées", len(raw_core)),
                ],
                columns=["Vérification", "Valeur"],
            ))
            """
        ),
        code(
            """
            key_cols = ["optimizer", "seed", "perturbation_type", "requested_severity"]
            merged_proof = raw_core.merge(
                proof_runs,
                on=key_cols,
                how="inner",
                suffixes=("_raw_saved", "_proof_rerun"),
            )
            assert len(merged_proof) == 650
            metric_comparison = pd.DataFrame(
                {
                    "metric": ["test_accuracy", "macro_f1", "loss"],
                    "max_abs_diff_saved_vs_proof_rerun": [
                        float((merged_proof[f"{metric}_raw_saved"] - merged_proof[f"{metric}_proof_rerun"]).abs().max())
                        for metric in ["test_accuracy", "macro_f1", "loss"]
                    ],
                }
            )
            display(metric_comparison)
            display(Markdown(
                "Cette comparaison vérifie que la grille de runs est la même entre le fichier brut final "
                "et la rerun complète avec historique. Les petites différences numériques éventuelles "
                "viennent de l'environnement d'exécution, mais la structure expérimentale est identique."
            ))
            """
        ),
        md("## 9. Recalcul des résultats agrégés"),
        code(
            """
            core_aggregate = aggregate_raw_results(raw_core).reset_index(drop=True)
            combined_raw = pd.read_csv(ROOT / "results/v2/aggregated/v2_raw_combined.csv")
            recomputed_combined = aggregate_raw_results(combined_raw).reset_index(drop=True)
            saved_aggregate = pd.read_csv(ROOT / "results/v2/aggregated/v2_aggregated_summary.csv")
            group_cols = [
                "dataset",
                "protocol",
                "robustness_setting",
                "optimizer",
                "perturbation_type",
                "requested_severity",
            ]
            compare_cols = list(recomputed_combined.columns)
            recomputed_combined = recomputed_combined.sort_values(group_cols).reset_index(drop=True)
            saved_aggregate = saved_aggregate.sort_values(group_cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(
                recomputed_combined[compare_cols],
                saved_aggregate[compare_cols],
                check_exact=False,
                atol=1e-9,
                rtol=1e-9,
            )
            display(Markdown(
                "Le fichier agrégé sauvegardé est recalculé exactement depuis `v2_raw_combined.csv` "
                "(1125 lignes). Le tableau ci-dessous montre l'agrégat pur du protocole principal Cora "
                "(650 lignes), pour éviter de mélanger les validations complémentaires avec le coeur du projet."
            ))
            display(core_aggregate.sort_values(["perturbation_type", "requested_severity", "mean_test_accuracy"], ascending=[True, True, False]).head(20))
            display(Markdown(
                "La moyenne seule ne suffit pas: avec 10 graines, on voit aussi l'écart-type et l'IC95. "
                "Si les intervalles d'Adam et RMSProp se recouvrent, il faut parler de performances proches "
                "et éviter une dominance universelle."
            ))
            """
        ),
        md("## 10. Vérification des validations complémentaires"),
        code(
            """
            validation_files = {
                "cross_dataset": ROOT / "results/v2/raw/v2_cross_dataset.csv",
                "tuned_protocol": ROOT / "results/v2/raw/v2_tuned_cora.csv",
                "inference_time_robustness": ROOT / "results/v2/raw/v2_inference_robustness.csv",
                "gradient_diagnostics_runs": ROOT / "results/v2/diagnostics/v2_optimizer_diagnostics_runs_clean.csv",
                "gradient_diagnostics_epochs": ROOT / "results/v2/diagnostics/v2_gradient_history_clean.csv",
            }
            validation_summary = []
            for label, path in validation_files.items():
                assert path.exists(), path
                frame = pd.read_csv(path)
                validation_summary.append(
                    {
                        "validation": label,
                        "fichier": str(path.relative_to(ROOT)),
                        "lignes": len(frame),
                        "colonnes": len(frame.columns),
                    }
                )
            display(pd.DataFrame(validation_summary))

            cross = pd.read_csv(validation_files["cross_dataset"])
            tuned = pd.read_csv(validation_files["tuned_protocol"])
            inference = pd.read_csv(validation_files["inference_time_robustness"])
            diagnostics = pd.read_csv(validation_files["gradient_diagnostics_runs"])

            display(cross.groupby(["dataset", "optimizer"]).size().reset_index(name="runs").head(15))
            display(tuned.groupby(["optimizer", "perturbation_type"]).size().reset_index(name="runs").head(15))
            display(inference.groupby(["optimizer", "perturbation_type"]).size().reset_index(name="runs").head(15))
            display(diagnostics.groupby(["optimizer"]).agg(runs=("seed", "nunique"), mean_gradient=("mean_gradient_l2_norm", "mean")).reset_index())
            """
        ),
        md(
            """
            **Ce que ces validations prouvent**

            - `cross_dataset`: le pipeline peut être exécuté sur Cora, CiteSeer et PubMed pour une sévérité contrôlée.
            - `tuned_protocol`: un protocole de tuning séparé existe pour comparer un réglage alternatif.
            - `inference_time_robustness`: on distingue entraîner sur graphe perturbé et tester un modèle propre sous perturbation.
            - `gradient_diagnostics`: les courbes de gradients/mémoire sont disponibles pour les runs clean.

            **Ce que ces validations ne prouvent pas**

            - Elles ne prouvent pas une robustesse universelle.
            - Elles ne remplacent pas les 650 runs Cora du protocole principal.
            - Les diagnostics de gradients ne couvrent pas toutes les perturbations; ils couvrent les runs clean.
            """
        ),
        md("## 11. Tests automatiques et CI"),
        code(
            """
            test_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py"))
            display(pd.DataFrame({"test_file": test_files}))
            print((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
            """
        ),
        code(
            """
            test_run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--disable-warnings"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=300,
            )
            print(test_run.stdout)
            if test_run.stderr:
                print(test_run.stderr)
            assert test_run.returncode == 0
            display(Markdown(
                "La CI vérifie qu'une modification future ne casse pas silencieusement le code, "
                "les tests, un smoke experiment ou le dashboard. Elle ne prouve pas à elle seule "
                "que toutes les conclusions scientifiques sont vraies."
            ))
            """
        ),
        md(
            """
            ## 12. Conclusion défendable

            - Under this fixed protocol on Cora, Adam and RMSProp are the strongest observed optimizers.
            - The difference between Adam and RMSProp is small and must not be described as universal dominance.
            - The conclusions are limited to this architecture, these datasets, these hyperparameters, and random perturbations.
            - The notebook provides direct traceability from code to raw results to final figures.
            """
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
    print(NOTEBOOK_PATH)
