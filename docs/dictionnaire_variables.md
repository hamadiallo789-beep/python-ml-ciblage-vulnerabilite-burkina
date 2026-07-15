# Dictionnaire des variables — `menages_pmt_burkina.csv`

Base de **1 100 ménages**, 25 variables. Une ligne = un ménage enquêté.

| Variable | Type | Description |
|---|---|---|
| `id_menage` | int | Identifiant unique du ménage |
| `region` | texte | Région d'enquête (Sahel, Est, Nord, Centre-Nord, Boucle du Mouhoun) |
| `milieu` | texte | Rural / Urbain |
| `taille_menage` | int | Nombre de personnes dans le ménage |
| `nb_enfants_moins_15` | int | Nombre d'enfants de moins de 15 ans |
| `ratio_dependance` | float | (Taille du ménage − actifs) / actifs — charge de dépendance |
| `sexe_chef_menage` | texte | Homme / Femme |
| `age_chef_menage` | int | Âge du chef de ménage (années) |
| `niveau_education_chef` | texte | Aucun / Primaire / Secondaire / Supérieur |
| `materiau_toit` | texte | Chaume-Paille / Tôle / Dur (béton-tuile) |
| `materiau_mur` | texte | Banco (terre) / Brique de ciment / Dur |
| `source_eau_potable` | texte | Puits non protégé / Puits protégé-Forage / Robinet-Eau courante |
| `possede_terre` | 0/1 | Le ménage possède-t-il une parcelle agricole |
| `superficie_terre_ha` | float | Superficie cultivée (hectares) |
| `nb_bovins` | int | Nombre de bovins possédés |
| `nb_petits_ruminants` | int | Nombre de moutons/chèvres |
| `nb_volailles` | int | Nombre de volailles |
| `possede_telephone` | 0/1 | Possession d'un téléphone |
| `possede_radio` | 0/1 | Possession d'une radio |
| `possede_velo` | 0/1 | Possession d'un vélo |
| `possede_moto` | 0/1 | Possession d'une moto |
| `distance_marche_km` | float | Distance au marché le plus proche (km) — 1,5% de valeurs manquantes |
| `score_consommation_alimentaire` | int | Score de Consommation Alimentaire (SCA, 0-112, standard PAM) |
| `depenses_mensuelles_pc_fcfa` | float | Dépenses mensuelles par tête (FCFA) — 2% de valeurs manquantes |
| `vulnerable` | 0/1 | **Variable cible** — 1 = ménage éligible à l'assistance |

## Variables dérivées créées lors de la préparation (script 01)

| Variable | Description |
|---|---|
| `educ_score` | Encodage ordinal du niveau d'éducation (0 à 3) |
| `toit_score` / `mur_score` / `eau_score` | Encodage ordinal des matériaux d'habitat et de la source d'eau (0 à 2) |
| `sexe_chef_femme` | Indicatrice binaire (1 = chef de ménage femme) |
| `milieu_rural` | Indicatrice binaire (1 = milieu rural) |
| `region_*` | Indicatrices one-hot de la région (référence : Boucle du Mouhoun) |

## Notes méthodologiques

- **Nature des données** : jeu de données **synthétique**, construit pour reproduire la structure d'une enquête de ciblage type Proxy Means Test (PMT), telle qu'utilisée par les agences humanitaires (PAM, HCR, ONG) pour identifier les ménages éligibles à une assistance (transferts monétaires, vivres).
- **Construction du label `vulnerable`** : un indice de vulnérabilité latent a été calculé comme combinaison pondérée des variables d'habitat, d'actifs, d'éducation, de consommation alimentaire et de contexte régional, additionné d'un bruit aléatoire substantiel — le problème de classification n'est donc pas trivial, comme en conditions réelles.
- **Prévalence** : 35,6% des ménages sont classés vulnérables, cohérent avec les taux d'insécurité alimentaire observés dans les régions Sahel/Est du Burkina Faso.
- **Valeurs manquantes** : injectées volontairement sur `distance_marche_km` et `depenses_mensuelles_pc_fcfa` pour refléter la non-réponse réelle en collecte de terrain.
