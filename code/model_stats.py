import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# 1. Wczytanie modelu i danych
try:
    with open('final_optimized_suicide_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('final_suicide_features_with_eda.csv')
    print("✅ Model i dane wczytane pomyślnie!")
except FileNotFoundError:
    print("❌ Błąd: Upewnij się, że pliki .pkl i .csv są w tym samym folderze co skrypt.")

# 2. Przygotowanie danych do wykresu
features = ['senior_pct', 'male_pct', 'substances_pct', 'youth_pct']
target = 'target_mortality_rate'
name_map = {
    'senior_pct': 'Udział seniorów (60+)',
    'male_pct': 'Udział mężczyzn',
    'substances_pct': 'Pod wpływem (alkohol/leki)',
    'youth_pct': 'Udział młodzieży (0-18)'
}

# Pobranie ważności i korelacji
importances = model.feature_importances_
correlations = [df[f].corr(df[target]) for f in features]

results = pd.DataFrame({
    'Feature': [name_map[f] for f in features],
    'Importance': importances,
    'Correlation': correlations
}).sort_values('Importance', ascending=False)

sns.set_theme(style="whitegrid")

# --- WYKRES 1: KIERUNKOWA ISTOTNOŚĆ
plt.figure(figsize=(12, 6))
# Sortujemy od najmniej do najbardziej ważnych dla barh
results_plot = results.sort_values('Importance', ascending=True)
colors = ['#d7191c' if c > 0 else '#2c7bb6' for c in results_plot['Correlation']]

bars1 = plt.barh(results_plot['Feature'], results_plot['Importance'], color=colors, alpha=0.8)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#d7191c', lw=6, label='Wzrost cechy = Wyższa śmiertelność (Ryzyko +)'),
    Line2D([0], [0], color='#2c7bb6', lw=6, label='Wzrost cechy = Niższa śmiertelność (Ochrona -)')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

# Etykiety %
for bar in bars1:
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.1%}', va='center', weight='bold', fontsize=11)

plt.title('Kierunkowy wpływ czynników na śmiertelność (Model)', fontsize=15, pad=20)
plt.xlim(0, results['Importance'].max() + 0.1)
plt.tight_layout()
plt.savefig('wykres_final_importance_directional.png', dpi=300)


# --- WYKRES 2: KLASYCZNY FEATURE IMPORTANCE ---
plt.figure(figsize=(10, 5))

# Używamy jednego, spójnego koloru 'teal'
bars2 = sns.barplot(
    data=results, 
    x='Importance', 
    y='Feature', 
    color='teal', 
    edgecolor='black',
    alpha=0.75
)

# Dodawanie wartości procentowych na końcach słupków
for p in bars2.patches:
    width = p.get_width()
    plt.text(
        width + 0.005, 
        p.get_y() + p.get_height()/2, 
        f'{width:.1%}', 
        va='center', 
        fontsize=11, 
        weight='bold'
    )

plt.title('Ranking ważności cech (Model Zoptymalizowany)', fontsize=14, pad=15)
plt.xlabel('Waga ważności (Importance)', fontsize=12)
plt.ylabel('Cecha', fontsize=12)
plt.xlim(0, results['Importance'].max() + 0.08) # Zapas na etykiety
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# Zapis ostatecznego wykresu
plt.savefig('wykres_final_importance_simple.png', dpi=300)

print("\n--- ANALIZA KIERUNKOWA ---")
for _, row in results.sort_values('Importance', ascending=False).iterrows():
    direction = "PODNOSI" if row['Correlation'] > 0 else "OBNIŻA"
    print(f"• {row['Feature']}: Odpowiada za {row['Importance']:.1%} decyzji modelu. Statystycznie {direction} śmiertelność.")