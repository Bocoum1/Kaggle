# Resultats et discussion

Les chiffres sont issus de l'execution locale,
pas du leaderboard Kaggle. Le [protocole](README.md) et les
[parametres enregistres](results/protocol.json) permettent de la reproduire.

## 1. La complexite ne garantit pas le meilleur ajustement hors entrainement

Ridge avec alpha = 100 obtient une RMSE logarithmique moyenne de 0,1328 sur
les cinq plis, contre 0,1399 pour le gradient boosting et 0,1485 pour la foret.
La reference constante obtient 0,4054.

La foret a pourtant une erreur d'entrainement bien plus basse : 0,0545,
contre 0,1178 pour Ridge. Cet ecart entrainement-validation est compatible
avec davantage de surapprentissage, sans demontrer une inferiorite generale
des forets. Les budgets et grilles explores sont limites et differents.

Les moyennes utilisees pour selectionner les parametres ne sont pas des
estimations non biaisees de la performance du modele selectionne.

## 2. Une bonne MAE peut coexister avec une mauvaise RMSE

Sur les 292 logements du test interne, Ridge obtient :

- RMSE sur log(prix) : **0,1798**, contre 0,3740 pour la reference.
- MAE : **20 645 $**, contre 51 929 $ pour la reference.
- RMSE en dollars : **110 805 $**, contre 80 557 $ pour la reference.
- R2 : **-0,982**, contre -0,048 pour la reference.

Le R2 negatif ne doit pas etre masque : les erreurs au carre depassent
celles d'une prediction constante a la moyenne du test. La reference du
projet est differente : elle utilise la moyenne des log-prix du developpement.

![Predictions et residus, sans suppression des erreurs extremes](results/residuals.png)

## 3. Un cas d'extrapolation a examiner

Le logement Id 1299 a un prix réel de 160 000 $ alors que le modèle prédit
2 015 662 $. À lui seul, ce logement représente 96,05 % de la somme des erreurs au carré sur le jeu de test
.
Sa surface habitable (GrLivArea) est de 5 642, alors que la plus grande surface dans les données d'entraînement est de 4 676.
Le modèle doit donc faire une prédiction sur une valeur qu'il n'a jamais rencontrée pendant l'entraînement.
Cet écart important peut s'expliquer par l'extrapolation du modèle linéaire sur le logarithme du prix, puis par le passage à l'échelle initiale avec l'exponentielle

L'erreur absolue médiane sur le test est de 8 883 $, et 95 % des logements ont une erreur inférieure ou égale à 48 848 $. Ces résultats permettent de mieux comprendre la répartition des erreurs, en complément de la RMSE. Tous les logements ont été conservés dans le calcul final.

Retirer la maison qui pose problème après avoir analysé les résultats du test fausserait l'évaluation du modèle. Il est donc préférable de la garder pour avoir une vision plus juste de ses performances.
## 4. Ce que l'on peut conclure
Sur ce découpage des données, Ridge est le modèle retenu par la validation. Il permet de réduire l'erreur absolue moyenne, mais reste sensible aux valeurs extrêmes, comme le montre le cas du logement Id 1299.

Un autre projet pourrait etudier les transformations des surfaces,
des modeles robustes et un choix multicritere. 
## Fichiers de preuve

- [Comparaison de validation](results/validation.csv).
- [Metriques de test](results/test-metrics.csv).
- [Protocole, versions et empreinte des donnees](results/protocol.json).
- Grilles completes, indices et predictions individuelles generes localement dans `artifacts/`.

Les donnees brutes et les predictions individuelles ne sont pas ajoutees au
depot. Les figures et tableaux agreges documentent l'experience sans republier
le jeu de donnees.
