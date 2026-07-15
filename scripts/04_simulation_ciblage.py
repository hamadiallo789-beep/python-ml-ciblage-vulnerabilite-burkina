"""
PROJET   : Ciblage humanitaire par Machine Learning (Proxy Means Test) — Burkina Faso
FICHIER  : 04_simulation_ciblage.py
OBJET    : Simulation d'un scenario operationnel de ciblage sous contrainte
           budgetaire : comparaison de 3 strategies (aleatoire, proxy simple,
           modele ML) sur leur capacite a effectivement couvrir les menages
           reellement vulnerables, avec calcul des erreurs d'inclusion et
           d'exclusion — les deux indicateurs cles utilises par les agences
           humanitaires (PAM, HCR, etc.) pour auditer un ciblage.
AUTEUR   : Hama DIALLO (Sekou)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("../output/predictions_test.csv")

# -----------------------------------------------------------------------------
# SCENARIO : le programme dispose d'un budget qui ne permet de couvrir que
# 30% des menages de la zone. Comment choisir qui recoit l'assistance ?
# -----------------------------------------------------------------------------
budget_pct = 0.30
n_test = len(df)
n_beneficiaires = int(round(n_test * budget_pct))
print(f"Menages evalues : {n_test}  |  Places disponibles (budget 30%) : {n_beneficiaires}")

rng = np.random.default_rng(7)

# A) Ciblage aleatoire — scenario "sans methode", loterie
idx_random = rng.choice(df.index, size=n_beneficiaires, replace=False)
selected_random = df.loc[idx_random]

# B) Proxy simple — ciblage "a l'oeil" sur un seul critere visible (toit)
df["rand_tiebreak"] = rng.random(len(df))
selected_proxy = df.sort_values(["toit_score", "rand_tiebreak"]).head(n_beneficiaires)

# C) Modele ML — score de probabilite de la regression logistique (script 03)
selected_ml = df.sort_values("proba_logit", ascending=False).head(n_beneficiaires)

def evaluer_ciblage(selected, all_data):
    total_vulnerables = all_data["y_true"].sum()
    vulnerables_couverts = selected["y_true"].sum()
    erreur_inclusion = (selected["y_true"] == 0).sum() / len(selected)
    erreur_exclusion = (total_vulnerables - vulnerables_couverts) / total_vulnerables
    return {
        "vulnerables_couverts": int(vulnerables_couverts),
        "total_vulnerables": int(total_vulnerables),
        "taux_couverture": round(vulnerables_couverts / total_vulnerables, 3),
        "erreur_inclusion": round(erreur_inclusion, 3),
        "erreur_exclusion": round(erreur_exclusion, 3),
    }

res_a = evaluer_ciblage(selected_random, df)
res_b = evaluer_ciblage(selected_proxy, df)
res_c = evaluer_ciblage(selected_ml, df)

print("\n=== RESULTATS DE LA SIMULATION (budget = 30% des menages) ===")
print("A) Ciblage aleatoire      :", res_a)
print("B) Proxy simple (toit)    :", res_b)
print("C) Modele ML (Logistique) :", res_c)

gain_vs_aleatoire = round((res_c["taux_couverture"] - res_a["taux_couverture"]) * 100, 1)
gain_vs_proxy = round((res_c["taux_couverture"] - res_b["taux_couverture"]) * 100, 1)
print(f"\nGain du modele ML vs ciblage aleatoire : +{gain_vs_aleatoire} points de couverture")
print(f"Gain du modele ML vs proxy simple       : +{gain_vs_proxy} points de couverture")

# -----------------------------------------------------------------------------
# GRAPHIQUE COMPARATIF
# -----------------------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
strategies = ["Aleatoire", "Proxy simple\n(toit)", "Modele ML\n(Regression Logistique)"]
couverture = [res_a["taux_couverture"], res_b["taux_couverture"], res_c["taux_couverture"]]
colors = ["#9e9e9e", "#fb8c00", "#2e7d32"]
bars = plt.bar(strategies, [c * 100 for c in couverture], color=colors)
for bar, val in zip(bars, couverture):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
              f"{val * 100:.1f}%", ha="center", fontweight="bold")
plt.ylabel("% des menages vraiment vulnerables effectivement couverts")
plt.title(f"Efficacite du ciblage selon la strategie (budget = {int(budget_pct * 100)}% des menages)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("../output/05_comparaison_ciblage.png")
plt.close()

print("\nGraphique exporte : output/05_comparaison_ciblage.png")

# Export du tableau de resultats
resultats = pd.DataFrame([
    {"strategie": "Aleatoire", **res_a},
    {"strategie": "Proxy simple (toit)", **res_b},
    {"strategie": "Modele ML (Regression Logistique)", **res_c},
])
resultats.to_csv("../output/resultats_simulation_ciblage.csv", index=False)
print("Tableau de resultats exporte : output/resultats_simulation_ciblage.csv")
