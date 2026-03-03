import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_ind

try:
    with open('dane.pkl', 'rb') as f:
        df = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")


# 2. Definicja okresu pandemii (2020-2021)
df['is_covid'] = df['Rok'].isin([2020, 2021]).astype(int)

# 3. Porównanie kluczowych wskaźników
metrics = ['target_mortality_rate', 'substances_pct', 'youth_pct', 'SES_instability_index']
comparison = df.groupby('is_covid')[metrics].mean()

print("Porównanie średnich (0 = przed/po, 1 = COVID):")
print(comparison)

# 4. Wykresy porównawcze
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Śmiertelność w czasie COVID
sns.boxplot(ax=axes[0], x='is_covid', y='target_mortality_rate', data=df, palette='Set2')
axes[0].set_title('Śmiertelność: COVID vs Reszta lat')
axes[0].set_xticklabels(['Inne lata', 'COVID (2020-21)'])

# Substancje w czasie COVID
sns.boxplot(ax=axes[1], x='is_covid', y='substances_pct', data=df, palette='Pastel1')
axes[1].set_title('Substancje: COVID vs Reszta lat')
axes[1].set_xticklabels(['Inne lata', 'COVID (2020-21)'])

plt.tight_layout()
plt.savefig('covid_impact_analysis.png')

# 5. Prosty test statystyczny (t-test) dla śmiertelności
group_covid = df[df['is_covid'] == 1]['target_mortality_rate']
group_other = df[df['is_covid'] == 0]['target_mortality_rate']
t_stat, p_val = ttest_ind(group_covid, group_other)

print(f"\nWynik testu t dla śmiertelności: p-value = {p_val:.4f}")

# Porównanie kluczowych wskaźników
covid_comparison = df.groupby('is_covid')[['target_mortality_rate', 'substances_pct', 'youth_pct']].mean()
print(covid_comparison)

# Wizualizacja różnicy w śmiertelności
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_covid', y='target_mortality_rate', data=df)
plt.title('Śmiertelność zamachów: Lata normalne vs COVID (2020-21)')
plt.savefig('covid_mortality_box.png')