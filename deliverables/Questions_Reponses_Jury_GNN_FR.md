# Questions - Reponses Jury GNN

## Pourquoi 200 epoques ?

Pour donner le meme budget d'entrainement a chaque optimiseur et eviter qu'un optimiseur profite d'un temps different.

## Pourquoi 10 graines ?

Pour estimer la variabilite aleatoire et calculer moyenne, ecart-type et IC95.

## Pourquoi 650 runs ?

Parce que le coeur de l'etude combine 5 optimiseurs, 13 conditions et 10 graines.

## Pourquoi Cora ?

Cora est un benchmark classique de classification de noeuds, assez petit pour une matrice reproductible complete.

## Pourquoi un GCN a deux couches ?

C'est l'architecture de reference la plus simple pour isoler l'effet de l'optimiseur.

## Pourquoi ces optimiseurs ?

Ils couvrent Adam, AdamW, RMSProp, AdaGrad et SGD, donc des familles adaptatives et non adaptatives.

## Pourquoi SGD est faible dans ce protocole ?

Sous ce taux d'apprentissage fixe et sans tuning principal, SGD converge moins bien. Ce n'est pas une critique generale de SGD.

## Pourquoi le masquage d'attributs ?

Masquage aléatoire d’une proportion de caractéristiques actives non nulles. Cette definition correspond au protocole final et garde les valeurs deja nulles intactes.

## Est-ce de la robustesse adversariale ?

Non. Les perturbations sont aleatoires, pas optimisees contre le modele.

## Pourquoi suppression et fausses aretes ?

Elles representent deux erreurs structurelles opposees: manque de citations et connexions artificielles.

## Perturbation a l'entrainement ou a l'inference ?

A l'entrainement, le modele apprend sur les donnees perturbees. A l'inference, il apprend proprement puis on perturbe seulement l'evaluation.

## Pourquoi les resultats changent selon les datasets ?

La densite, les attributs et les classes different, donc le signal de propagation change.

## Comment garantir la reproductibilite ?

Graines fixes, configs versionnees, identifiant de version, metadonnees materiel, resultats bruts et tests automatises.

## Ou sont les resultats bruts ?

Dans le depot, separes des agregats, avec une ligne par entrainement reel.

## Pourquoi ne pas dire qu'Adam est universellement meilleur ?

Parce que l'etude teste un dataset, une architecture et un protocole precis.

## Pourquoi les intervalles de confiance ?

Ils montrent l'incertitude due aux graines et evitent de surinterpreter de petites differences.

## Le projet a-t-il ete entraine sur Google Colab ?

Les experiences principales ont ete lancees localement sur CPU; la reproductibilite compte plus que la plateforme cloud.

## Prochaines etapes concretes ?

Tester d'autres architectures, ajouter des attaques adversariales reelles, et elargir les datasets.
