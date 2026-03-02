import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ładowanie danych
try:
    with open('dane.pkl', 'rb') as f:
        df = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")

# 1. Wczytanie danych przetworzonych
# Zakładamy, że plik 'final_suicide_features.csv' jest w tym samym folderze
df = pd.read_csv('final_suicide_features.csv')

# 2. Agregacja danych do średnich krajowych na rok
df_annual = df.groupby('Rok').mean(numeric_only=True).reset_index()

# Ustawienie stylu wizualnego
sns.set_theme(style="whitegrid")

# --- WYKRES 1: ZMIANY DEMOGRAFICZNE ---
plt.figure(figsize=(12, 6))
# plt.plot(df_annual['Rok'], df_annual['male_pct'], marker='o', label='Mężczyźni (%)', linewidth=2, color='#0055CC')
plt.plot(df_annual['Rok'], df_annual['youth_pct'], marker='s', label='Młodzież 0-18 (%)', linestyle='--', color='#CC8800')
plt.plot(df_annual['Rok'], df_annual['young_adult_pct'], marker='^', label='Młodzi dorośli 19-34 (%)', linestyle='--', color='#009966')
plt.plot(df_annual['Rok'], df_annual['middle_age_pct'], marker='d', label='Wiek średni 35-59 (%)', linestyle='--', color='#CC5500')
plt.plot(df_annual['Rok'], df_annual['senior_pct'], marker='v', label='Seniorzy 60+ (%)', linestyle='--', color='#CC77AA')

plt.title('Zmiany demograficzne w zamachach samobójczych (Średnie roczne)', fontsize=14)
plt.xlabel('Rok', fontsize=12)
plt.ylabel('Udział w ogólnej liczbie zamachów', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('wykres_demografia.png', dpi=300)

# --- WYKRES 2: CZYNNIKI ZEWNĘTRZNE I SOCJOEKONOMICZNE ---
plt.figure(figsize=(12, 6))
plt.plot(df_annual['Rok'], df_annual['substances_pct'], marker='o', label='Pod wpływem substancji (%)', color='orange', linewidth=2)
plt.plot(df_annual['Rok'], df_annual['SES_instability_index'], marker='x', label='Indeks niestabilności SES', color='red', linewidth=2)
plt.plot(df_annual['Rok'], df_annual['institution_contact_pct'], marker='s', label='Kontakt z instytucjami (%)', color='green')
plt.plot(df_annual['Rok'], df_annual['weekend_pct'], marker='*', label='Zamachy w weekendy (%)', color='purple')
plt.plot(df_annual['Rok'], df_annual['education_higher_pct'], marker='D', label='Wykształcenie wyższe (%)', color='blue')

plt.title('Czynniki zewnętrzne i socjoekonomiczne (Średnie roczne)', fontsize=14)
plt.xlabel('Rok', fontsize=12)
plt.ylabel('Wartość wskaźnika', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('wykres_czynniki_zewnetrzne.png', dpi=300)



print("Wykresy zostały wygenerowane i zapisane.")