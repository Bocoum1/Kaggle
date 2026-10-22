# House Prices : evaluer la generalisation en regression

## Question

Les modeles non lineaires apportent-ils un gain par rapport a une regression
Ridge pour predire le prix d'un logement, avec les memes informations et le
meme protocole de validation ?

Cette etude prolonge le [notebook original](../regression.ipynb), conserve sans
modification. Elle corrige en particulier l'imputation avant decoupage et
remplace les scores d'entrainement seuls par une evaluation hors entrainement.

## Donnees et protocole

Telecharger `train.csv` depuis [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
et le placer dans `data/house-prices/train.csv`. Les donnees ne sont pas versionnees.
Le `test.csv` Kaggle ne contient pas SalePrice : ce n'est pas notre test interne.

- Separation aleatoire 80 % developpement / 20 % test interne, graine 20261015.
- Validation croisee a 5 plis identiques sur le developpement pour toutes les configurations.
- Id exclu ; MSSubClass traite comme une categorie ; aucune suppression basee sur SalePrice.
- Imputation mediane et standardisation dans chaque pli ; encodage des categories avec gestion des inconnues.
- Cible : logarithme naturel du prix. Critere de selection : RMSE moyenne sur log(prix).
- Reference : moyenne des log-prix d'entrainement, soit une moyenne geometrique apres retour en dollars.
- Ridge : alpha dans {0.1, 1, 10, 100}.
- Foret : 200 arbres ; min_samples_leaf dans {1, 3} ; max_features dans {0.7, 1.0}.
- Gradient boosting : 200 arbres, learning_rate 0.05, min_samples_leaf 3 ; max_depth dans {2, 3}.
- Le modele retenu est reajuste sur tout le developpement ; seul ce modele et la reference passent au test.

Toutes les familles partagent le meme pretraitement pour cette premiere
comparaison. Standardiser n'est pas necessaire aux arbres, mais evite de
changer plusieurs elements du protocole a la fois. La grille est
petite ; il ne s'agit pas d'optimiser un classement Kaggle.

## Reproduire

Depuis la racine du depot :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-study.txt
python -m house_prices.experiment train
python -m house_prices.experiment evaluate
python -m unittest discover -s tests
```

Le script est commente en francais. La premiere commande selectionne sans
evaluer le test ; la seconde calcule RMSE(log), MAE et RMSE en dollars, et R2.
Les versions, empreintes SHA-256, indices, plis et grilles sont enregistres
dans `artifacts/house-prices/`. Les modeles et donnees restent locaux.
Ne pas charger de fichier joblib provenant d'une source non fiable.

## Lire les resultats avec prudence

Les moyennes de validation servent a la selection : leurs scores sont donc
optimistes apres choix des parametres. L'ecart-type entre plis n'est pas un
intervalle de confiance, et les plis ne sont pas des repetitions independantes.
Le test interne sert a l'evaluation finale, pas a reprendre le reglage.

Le retour exponentiel des predictions ne corrige pas le biais de transformation
et ne vise pas directement la moyenne conditionnelle du prix en dollars.
Les metriques monetaires sont donc complementaires au critere optimise.

Les donnees avaient deja ete explorees dans le notebook original : le test
est reserve dans cette nouvelle procedure, pas totalement inedit dans l'histoire
du projet. Un decoupage aleatoire ne garantit pas la generalisation a une autre
ville, une autre periode ou une autre population. Les residus sont descriptifs,
pas une preuve causale sur les determinants du prix.

## References

- [Prevenir les fuites de donnees](https://scikit-learn.org/stable/common_pitfalls.html).
- [Pipelines et pretraitements composites](https://scikit-learn.org/1.7/modules/compose.html).
