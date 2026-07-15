"""
PROJET   : Ciblage humanitaire par Machine Learning (Proxy Means Test) — Burkina Faso
FICHIER  : 01_preparation_donnees.py
OBJET    : Import, controle qualite, imputation des valeurs manquantes et
           encodage des variables pour la modelisation.
AUTEUR   : Hama DIALLO (Sekou)
"""
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. IMPORT
# -----------------------------------------------------------------------------
df = pd.read_csv("../data/menages_pmt_burkina.csv")

print("Dimensions du jeu de donnees :", df.shape)
print("\nApercu :")
print(df.head())

# -----------------------------------------------------------------------------
# 2. CONTROLE QUALITE
# -----------------------------------------------------------------------------
print("\n--- Valeurs manquantes par colonne ---")
na_counts = df.isna().sum()
print(na_counts[na_counts > 0])

print("\n--- Prevalence de la variable cible (vulnerable) ---")
print(df["vulnerable"].value_counts(normalize=True).round(3))

print("\n--- Statistiques descriptives (variables numeriques) ---")
print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(2))

# Doublons eventuels sur id_menage
n_doublons = df["id_menage"].duplicated().sum()
print(f"\nDoublons id_menage : {n_doublons}")

# -----------------------------------------------------------------------------
# 3. IMPUTATION DES VALEURS MANQUANTES
# -----------------------------------------------------------------------------
# distance_marche_km et depenses_mensuelles_pc_fcfa contiennent quelques
# valeurs manquantes (non-reponse terrain, ~1.5-2%). Imputation par la mediane,
# une approche simple et robuste adaptee a des variables asymetriques.
df_clean = df.copy()
for col in ["distance_marche_km", "depenses_mensuelles_pc_fcfa"]:
    mediane = df_clean[col].median()
    n_manquant = df_clean[col].isna().sum()
    df_clean[col] = df_clean[col].fillna(mediane)
    print(f"Imputation {col}: {n_manquant} valeurs remplacees par la mediane ({mediane:.0f})")

# -----------------------------------------------------------------------------
# 4. ENCODAGE DES VARIABLES CATEGORIELLES ORDINALES
# -----------------------------------------------------------------------------
# Les variables d'habitat/education suivent un ordre naturel de "confort" —
# on les encode en scores ordinaux plutot qu'en dummies pures, ce qui est
# la pratique standard des modeles PMT (Proxy Means Test).
educ_map = {"Aucun": 0, "Primaire": 1, "Secondaire": 2, "Superieur": 3}
toit_map = {"Chaume/Paille": 0, "Tole": 1, "Dur (beton/tuile)": 2}
mur_map = {"Banco (terre)": 0, "Brique de ciment": 1, "Dur": 2}
eau_map = {"Puits non protege": 0, "Puits protege/Forage": 1, "Robinet/Eau courante": 2}

df_clean["educ_score"] = df_clean["niveau_education_chef"].map(educ_map)
df_clean["toit_score"] = df_clean["materiau_toit"].map(toit_map)
df_clean["mur_score"] = df_clean["materiau_mur"].map(mur_map)
df_clean["eau_score"] = df_clean["source_eau_potable"].map(eau_map)
df_clean["sexe_chef_femme"] = (df_clean["sexe_chef_menage"] == "Femme").astype(int)
df_clean["milieu_rural"] = (df_clean["milieu"] == "Rural").astype(int)

# One-hot encoding pour la region (variable nominale, sans ordre naturel)
region_dummies = pd.get_dummies(df_clean["region"], prefix="region", drop_first=True)
df_clean = pd.concat([df_clean, region_dummies], axis=1)

print("\nColonnes apres encodage :", df_clean.shape[1])

# -----------------------------------------------------------------------------
# 5. EXPORT DU JEU DE DONNEES PREPARE
# -----------------------------------------------------------------------------
df_clean.to_csv("../data/menages_pmt_burkina_prepare.csv", index=False)
print("\nFichier prepare exporte : data/menages_pmt_burkina_prepare.csv")
