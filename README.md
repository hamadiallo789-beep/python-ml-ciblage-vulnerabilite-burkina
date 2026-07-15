# Ciblage humanitaire par Machine Learning — Proxy Means Test (Burkina Faso)

Modèle de **Machine Learning en Python** (scikit-learn) pour prédire la vulnérabilité des ménages et optimiser le ciblage d'un programme d'assistance humanitaire (transferts monétaires / sécurité alimentaire) sous contrainte budgétaire. Ce dépôt sert de démonstration de compétences en data science appliquée au suivi-évaluation (MEAL) : modélisation prédictive, validation de modèle, et traduction des résultats en recommandation opérationnelle chiffrée.

## Contexte

Les programmes d'assistance humanitaire disposent rarement d'un budget suffisant pour couvrir l'ensemble d'une population en besoin. Le **Proxy Means Test (PMT)** est la méthode standard utilisée par les agences (PAM, HCR, ONG) pour estimer la vulnérabilité d'un ménage à partir de caractéristiques observables (habitat, actifs, éducation, taille du ménage) sans avoir à mesurer directement le revenu — difficile à collecter de façon fiable en zone rurale.

Ce projet simule un scénario réaliste : une enquête auprès de **1 100 ménages** dans 5 régions du Burkina Faso (Sahel, Est, Nord, Centre-Nord, Boucle du Mouhoun), avec pour objectif de prédire quels ménages sont réellement vulnérables et d'évaluer combien un modèle de ciblage bien calibré peut améliorer l'efficacité de l'assistance par rapport à un ciblage naïf.

> Les données utilisées sont **synthétiques**, construites pour reproduire la structure et les corrélations réelles d'une enquête PMT (le label de vulnérabilité intègre un bruit aléatoire volontaire pour que le problème de classification reste réaliste et non trivial).

## Objectifs

1. Prédire la vulnérabilité d'un ménage à partir de variables socio-économiques facilement observables sur le terrain.
2. Comparer une approche interprétable (régression logistique) et une approche non-linéaire (Random Forest), avec validation croisée.
3. Identifier les variables les plus déterminantes du risque de vulnérabilité.
4. **Chiffrer l'impact opérationnel** : simuler un ciblage sous contrainte budgétaire et comparer son efficacité à un ciblage aléatoire et à un proxy simple à un seul critère.

## Méthodologie et outils

- **Langage** : Python (scripts `.py` reproductibles, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)
- **Modèles** : Régression Logistique (baseline interprétable) et Random Forest (400 arbres), split train/test stratifié 75/25, validation croisée 5-fold
- **Évaluation** : Accuracy, Précision, Rappel, F1, AUC-ROC, matrice de confusion
- **Volet opérationnel** : simulation de ciblage budgétaire avec calcul des erreurs d'inclusion et d'exclusion — les deux indicateurs qu'utilisent les agences humanitaires pour auditer un ciblage

## Structure du dépôt

```
├── data/
│   ├── menages_pmt_burkina.csv              # Jeu de données brut (1 100 ménages, 25 variables)
│   └── menages_pmt_burkina_prepare.csv      # Version imputée/encodée (générée par le script 01)
├── scripts/
│   ├── 01_preparation_donnees.py            # Import, contrôle qualité, imputation, encodage
│   ├── 02_analyse_exploratoire.py           # Statistiques descriptives et visualisations
│   ├── 03_modelisation_ml.py                # Régression Logistique + Random Forest, ROC, importance
│   └── 04_simulation_ciblage.py             # Simulation de ciblage budgétaire (3 stratégies)
├── output/
│   ├── 01_courbe_roc.png … 05_comparaison_ciblage.png
│   ├── tableau_resume_par_region.csv
│   ├── predictions_test.csv
│   └── resultats_simulation_ciblage.csv
└── docs/
    └── dictionnaire_variables.md            # Dictionnaire complet des variables
```

## Comment exécuter

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
cd scripts
python 01_preparation_donnees.py
python 02_analyse_exploratoire.py
python 03_modelisation_ml.py
python 04_simulation_ciblage.py
```

Les scripts s'exécutent dans l'ordre depuis le dossier `scripts/` (chemins relatifs). Les graphiques et tableaux sont exportés automatiquement dans `output/`.

## Principaux résultats

**Prévalence de la vulnérabilité** : 35,6% des ménages (la plus élevée en région Sahel : 43%, contre 27% en Boucle du Mouhoun).

### Performance des modèles (jeu de test, n=275)

| Modèle | Accuracy | Précision | Rappel | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Régression Logistique | 82,9% | 79,3% | 70,4% | 74,6% | **0,885** |
| Random Forest | 79,6% | 73,9% | 66,3% | 69,9% | 0,865 |

Validation croisée 5-fold (Random Forest) : AUC moyen = 0,846 (écart-type 0,028) — performance stable et non due au hasard du découpage train/test.

**Modèle retenu : Régression Logistique** — meilleur AUC et totalement interprétable via ses coefficients, un atout décisif en contexte humanitaire où un ciblage doit pouvoir être expliqué et audité.

**Variables les plus prédictives** (Random Forest) : matériau du toit, dépenses mensuelles par tête, ratio de dépendance, source d'eau potable, matériau des murs, score de consommation alimentaire, distance au marché. Cohérent avec la littérature PMT : l'habitat reste le meilleur proxy de richesse en Afrique de l'Ouest.

### Simulation de ciblage (budget = 30% des ménages, n=275)

| Stratégie | Ménages vulnérables couverts | Taux de couverture | Erreur d'inclusion | Erreur d'exclusion |
|---|---|---|---|---|
| Aléatoire | 26 / 98 | 26,5% | 68,3% | 73,5% |
| Proxy simple (toit uniquement) | 42 / 98 | 42,9% | 48,8% | 57,1% |
| **Modèle ML (Régression Logistique)** | **67 / 98** | **68,4%** | **18,3%** | **31,6%** |

**Le modèle de Machine Learning couvre 68,4% des ménages réellement vulnérables avec le même budget qu'un ciblage aléatoire (+41,9 points) ou qu'un ciblage au jugé sur un seul critère (+25,5 points).** L'erreur d'inclusion (assistance donnée à un ménage non-vulnérable) chute de 68% à 18% — une amélioration directement traduisible en ressources économisées ou réorientées vers davantage de bénéficiaires réellement dans le besoin.

## Compétences démontrées

- Construction d'un pipeline complet de Machine Learning : préparation des données, encodage, split stratifié, entraînement, validation croisée
- Comparaison rigoureuse de modèles (interprétabilité vs performance) et justification du choix final
- Évaluation de classification déséquilibrée (accuracy, précision, rappel, F1, AUC-ROC, matrice de confusion)
- Traduction d'un modèle prédictif en **recommandation opérationnelle chiffrée** — capacité clé pour un poste MEAL/data science en contexte humanitaire
- Visualisation de données avec `matplotlib`/`seaborn` (courbes ROC, importance des variables, comparaisons de scénarios)
- Compréhension des méthodologies de ciblage humanitaire (Proxy Means Test, erreurs d'inclusion/exclusion)

## À propos

**Hama DIALLO (Sékou)** — Data Analyst / Chargé MEAL, spécialisé en suivi-évaluation de projets de développement et statistiques appliquées au Sahel.
Contact : hamadiallo789@gmail.com · Portfolio : [hamadiallo789-beep.github.io/portfolio](https://hamadiallo789-beep.github.io/portfolio)
