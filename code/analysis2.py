import pandas as pd
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

# --- WYKRES 1: STRUKTURA WIEKU ---
plt.figure(figsize=(12, 6))
age_cols = ['youth_pct', 'young_adult_pct', 'middle_age_pct', 'senior_pct']
plt.stackplot(df_annual['Rok'], [df_annual[c] for c in age_cols], 
              labels=['Młodzież', 'Młodzi dorośli', 'Wiek średni', 'Seniorzy'], alpha=0.7)

plt.title('Struktura Wieku Osób Podejmujących Zamachy Samobójcze (Średnie roczne)', fontsize=14)
plt.xlabel('Rok', fontsize=12)
plt.ylabel('Udział w ogólnej liczbie zamachów', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('wykres_demografia.png', dpi=300)

# --- WYKRES 2: EDUKACJA ---
plt.figure(figsize=(12, 6))
plt.plot(df_annual['Rok'], df_annual['edu_low_pct'], marker='o', label='Wykształcenie podstawowe (%)', color='orange', linewidth=2)
plt.plot(df_annual['Rok'], df_annual['edu_mid_pct'], marker='x', label='Wykształcenie średnie (%)', color='red', linewidth=2)
plt.plot(df_annual['Rok'], df_annual['edu_higher_pct'], marker='s', label='Wykształcenie wyższe (%)', color='green')
#plt.plot(df_annual['Rok'], df_annual['edu_unknown_pct'], marker='*', label='Wykształcenie - brak danych (%)', color='purple')

plt.title('Trend udziału w zamachach samobójczych w zależności od wykształcenia (2017-2025)', fontsize=14)
plt.xlabel('Rok', fontsize=12)
plt.ylabel('Wartość wskaźnika', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('wykres_edukacja.png', dpi=300)


# --- WYKRES 3: AREA CHART DLA PŁCI
plt.figure(figsize=(10, 6))
plt.fill_between(df_annual['Rok'], df_annual['male_pct'], label='Mężczyźni', color='skyblue', alpha=0.6)
plt.fill_between(df_annual['Rok'], 1 - df_annual['female_pct'], 1, label='Kobiety', color='pink', alpha=0.6)
plt.title('Trend udziału płci w zamachach samobójczych (2017-2025)', fontsize=14)
plt.ylabel('Udział procentowy', fontsize=12)
plt.ylim(0, 1)
plt.legend(loc='lower left')
plt.savefig('wykres_plec.png', dpi=300)

# --- WYKRES 4: Trend Śmiertelności ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_annual, x='Rok', y='target_mortality_rate', marker='o', color='red')
plt.ylim(0, 1)
plt.title('Trend Śmiertelności Zamachów Samobójczych w Polsce (Średnia Wojewódzka)')
plt.ylabel('Skuteczność prób (Zgony / Ogółem)')
plt.savefig('wykres_smiertelnosc_srednia.png', dpi=300)


print("Wykresy zostały wygenerowane i zapisane.")