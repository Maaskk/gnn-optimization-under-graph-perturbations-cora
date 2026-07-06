# Script de soutenance - GNN Cora

## 1. Mohamed Amine Kar-any - Slides 1-2 - environ 2 minutes

Bonjour. Aujourd'hui, nous presentons notre etude sur la robustesse des GNN sous perturbations aleatoires. La question est simple: si le graphe Cora perd des aretes, gagne de fausses aretes, ou perd une partie de ses attributs actifs, quel optimiseur garde l'entrainement le plus stable?

Transition: Hamza va maintenant presenter le dataset, le modele et les optimiseurs compares.

Mots a accentuer: perturbations aleatoires; comparaison controlee; Cora.

A ne pas dire: robustesse adversariale; meilleur optimiseur universel.

## 2. Hamza Elhaddaji - Slides 3-4 - environ 2.5 minutes

Cora est un reseau de citations: les noeuds sont des documents, les aretes representent des citations, et les attributs de noeuds sont des caracteristiques textuelles binaires. Nous utilisons un GCN a deux couches avec 16 canaux caches, dropout 0.5 et une perte cross-entropy.

Nous comparons Adam, AdamW, RMSProp, AdaGrad et SGD. Le but n'est pas de favoriser un optimiseur, mais de les placer dans le meme cadre experimental.

Transition: Ossama va detailler le protocole scientifique.

Mots a accentuer: meme architecture; meme dataset; cinq optimiseurs.

A ne pas dire: SGD est mauvais en general.

## 3. Ossama Ashad - Slides 5-7 - environ 3 minutes

Nous avons utilise 200 epoques par entrainement. Le budget fixe de 200 epoques assure une comparaison controlee. Le coeur de l'etude comporte 650 entrainements reels: 5 optimiseurs x 13 conditions x 10 graines.

Les graines permettent d'obtenir moyenne, ecart-type et IC95%. Le masquage d'attributs suit la definition: masquage aleatoire d'une proportion de caracteristiques actives non nulles. Pour la structure, nous supprimons des aretes ou nous ajoutons de fausses aretes sans self-loops et sans duplicats. Il s'agit de perturbations aleatoires, pas d'attaques adversariales.

Les conclusions sont limitees au dataset, a l'architecture et au protocole etudies.

Transition: Iliass va maintenant interpreter les resultats principaux.

Mots a accentuer: 650 entrainements reels; 200 epoques; IC95%; perturbations aleatoires.

A ne pas dire: resultats inventes; test utilise pour choisir les hyperparametres.

## 4. Iliass Ouchida - Slides 8-11 - environ 3 minutes

Sur le graphe propre, les optimiseurs adaptatifs obtiennent les meilleurs scores moyens. Sur les perturbations, Adam et RMSProp restent proches dans plusieurs conditions. Quand les intervalles se recouvrent, nous evitons de parler de dominance.

Le masquage d'attributs teste la perte d'information dans les attributs. Les perturbations structurelles testent la sensibilite aux connexions du graphe. SGD sous-performe ici dans le protocole fixe, mais cela ne veut pas dire que SGD est faible dans tous les contextes.

Transition: Mouhcine va presenter les validations complementaires, les limites et la demonstration.

Mots a accentuer: intervalles; prudence; protocole fixe.

A ne pas dire: Adam domine globalement.

## 5. Mouhcine Ayar - Slides 12-14 - environ 3 minutes

Nous avons ajoute trois validations: cross-dataset, protocole tune avec selection validation uniquement, et perturbation a l'inference apres entrainement propre. Ces validations aident a separer ce qui depend de Cora, de l'optimiseur, et du moment ou la perturbation est appliquee.

Les limites sont importantes: ce ne sont pas des attaques adversariales, les temps CPU dependent du materiel, et les conclusions restent liees au GCN a deux couches.

Je termine avec la demonstration du dashboard. Le site montre les resultats, les fichiers bruts, les diagnostics et les artefacts reproductibles.

Mots a accentuer: validations complementaires; limites honnetes; dashboard.

A ne pas dire: preuve universelle; inference live dans l'animation.
