import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ładowanie danych
try:
    with open('dane_eda.pkl', 'rb') as f:
        df_eda = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")

# Agregacja danych do średnich krajowych na rok
df_annual = df_eda.groupby('Rok').mean(numeric_only=True).reset_index()

sns.set_theme(style="whitegrid")

# --- WYKRES 1: STRUKTURA WIEKU (Stackplot) ---
plt.figure(figsize=(14, 7))
age_cols = ['youth_pct', 'young_adult_pct', 'middle_age_pct', 'senior_pct']
labels = ['Młodzież', 'Młodzi dorośli', 'Wiek średni', 'Seniorzy']
y_values = [df_annual[c] for c in age_cols]

# Rysowanie stackplot
plt.stackplot(df_annual['Rok'], y_values, labels=labels, alpha=0.7)

# Dodawanie etykiet do środka warstw
y_stack = np.cumsum(y_values, axis=0)
for i, col in enumerate(age_cols):
    for idx, row in df_annual.iterrows():
        # Obliczanie środka warstwy
        prev_y = y_stack[i-1][idx] if i > 0 else 0
        current_y = y_stack[i][idx]
        mid_y = (prev_y + current_y) / 2
        val = row[col]
        if val > 0.03: # Wyświetlaj tylko jeśli udział > 3% (czytelność)
            plt.text(row['Rok'], mid_y, f'{val:.1%}', ha='center', va='center', fontsize=9, weight='bold')

plt.title('Struktura Wieku Osób Podejmujących Zamachy Samobójcze', fontsize=14)
plt.ylabel('Udział w ogólnej liczbie', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('wykres_demografia.png', dpi=300)

# --- WYKRES 2: EDUKACJA ---
plt.figure(figsize=(12, 6))
edu_mapping = [
    ('edu_low_pct', 'Podstawowe', 'orange'),
    ('edu_mid_pct', 'Średnie', 'red'),
    ('edu_higher_pct', 'Wyższe', 'green')
]

for col, label, color in edu_mapping:
    plt.plot(df_annual['Rok'], df_annual[col], marker='o', label=label, color=color, linewidth=2)
    # Dodawanie etykiet nad punktami
    for x, y in zip(df_annual['Rok'], df_annual[col]):
        plt.text(x, y + 0.01, f'{y:.1%}', ha='center', color=color, weight='bold', fontsize=9)

plt.title('Trend wykształcenia w zamachach samobójczych (2017-2025)', fontsize=14)
plt.ylabel('Udział procentowy', fontsize=12)
plt.legend(loc='upper right')
plt.ylim(0, df_annual[['edu_low_pct', 'edu_mid_pct', 'edu_higher_pct']].values.max() + 0.1)
plt.savefig('wykres_edukacja.png', dpi=300)


# --- WYKRES 3: AREA CHART DLA PŁCI
plt.figure(figsize=(10, 6))
plt.fill_between(df_annual['Rok'], 0, df_annual['male_pct'], label='Mężczyźni', color='skyblue', alpha=0.6)
plt.fill_between(df_annual['Rok'], df_annual['male_pct'], 1, label='Kobiety', color='pink', alpha=0.6)

# Etykiety dla płci (środek obszarów)
for idx, row in df_annual.iterrows():
    # Etykieta dla mężczyzn
    plt.text(row['Rok'], row['male_pct']/2, f'{row["male_pct"]:.1%}', ha='center', va='center', fontsize=9, color='darkblue', weight='bold')
    # Etykieta dla kobiet
    plt.text(row['Rok'], (row['male_pct'] + 1)/2, f'{row["female_pct"]:.1%}', ha='center', va='center', fontsize=9, color='darkred', weight='bold')

plt.title('Trend udziału płci w zamachach samobójczych (2017-2025)', fontsize=14)
plt.ylim(0, 1.05)
plt.legend(loc='lower left')
plt.savefig('wykres_plec.png', dpi=300)

# --- WYKRES 4: Trend Śmiertelności ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_annual, x='Rok', y='target_mortality_rate', marker='o', color='red', linewidth=2.5)

# Dodawanie etykiet śmiertelności
for x, y in zip(df_annual['Rok'], df_annual['target_mortality_rate']):
    plt.text(x, y + 0.015, f'{y:.2f}', ha='center', va='bottom', color='red', weight='bold', fontsize=10)

plt.title('Trend Śmiertelności Zamachów Samobójczych (Zgony / Ogółem)', fontsize=14)
plt.ylabel('Skuteczność prób', fontsize=12)
plt.tight_layout()
plt.savefig('wykres_smiertelnosc_srednia.png', dpi=300)


print("Wykresy zostały wygenerowane i zapisane.")