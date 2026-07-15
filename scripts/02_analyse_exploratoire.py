"""
PROJET   : Ciblage humanitaire par Machine Learning (Proxy Means Test) — Burkina Faso
FICHIER  : 02_analyse_exploratoire.py
OBJET    : Statistiques descriptives et visualisations exploratoires (EDA)
           avant modelisation.
AUTEUR   : Hama DIALLO (Sekou)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("../data/menages_pmt_burkina.csv")

# -----------------------------------------------------------------------------
# 1. VUE D'ENSEMBLE
# -----------------------------------------------------------------------------
print("Prevalence globale de vulnerabilite :", df["vulnerable"].mean().round(3))
print("\nPrevalence par region :")
print(df.groupby("region")["vulnerable"].mean().sort_values(ascending=False).round(3))
print("\nPrevalence par milieu :")
print(df.groupby("milieu")["vulnerable"].mean().round(3))
print("\nPrevalence par sexe du chef de menage :")
print(df.groupby("sexe_chef_menage")["vulnerable"].mean().round(3))

# -----------------------------------------------------------------------------
# 2. GRAPHIQUE : distribution du score de consommation alimentaire (SCA)
#    selon le statut de vulnerabilite
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df, x="vulnerable", y="score_consommation_alimentaire",
    hue="vulnerable", palette=["#66bb6a", "#e53935"], legend=False
)
plt.xticks([0, 1], ["Non vulnerable", "Vulnerable"])
plt.xlabel("")
plt.ylabel("Score de consommation alimentaire (SCA)")
plt.title("SCA selon le statut de vulnerabilite")
plt.tight_layout()
plt.savefig("../output/04_sca_par_vulnerabilite.png")
plt.close()

# -----------------------------------------------------------------------------
# 3. TABLEAU DE SYNTHESE : materiau du toit x vulnerabilite
# -----------------------------------------------------------------------------
tab_toit = pd.crosstab(df["materiau_toit"], df["vulnerable"], normalize="index").round(3)
print("\n--- Taux de vulnerabilite par materiau de toit ---")
print(tab_toit)

# -----------------------------------------------------------------------------
# 4. CORRELATION ENTRE VARIABLES NUMERIQUES CLES ET LA VULNERABILITE
# -----------------------------------------------------------------------------
num_cols = [
    "taille_menage", "ratio_dependance", "superficie_terre_ha", "nb_bovins",
    "distance_marche_km", "score_consommation_alimentaire",
    "depenses_mensuelles_pc_fcfa", "vulnerable"
]
corr = df[num_cols].corr(numeric_only=True)
print("\n--- Correlations avec la cible 'vulnerable' ---")
print(corr["vulnerable"].sort_values(ascending=False).round(3))

# -----------------------------------------------------------------------------
# Resume exporte en CSV pour le README / reporting
# -----------------------------------------------------------------------------
resume = df.groupby("region").agg(
    nb_menages=("id_menage", "count"),
    taux_vulnerabilite=("vulnerable", "mean"),
    sca_moyen=("score_consommation_alimentaire", "mean"),
).round(2).sort_values("taux_vulnerabilite", ascending=False)
resume.to_csv("../output/tableau_resume_par_region.csv")
print("\nTableau de synthese exporte : output/tableau_resume_par_region.csv")
print(resume)
