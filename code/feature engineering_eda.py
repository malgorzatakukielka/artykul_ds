import pandas as pd
import numpy as np
import pickle
import os

# --- KONFIGURACJA ŚCIEŻEK ---
paths = {
    "wiek_dni": "source_data/zamachy_samobojcze_grupa_wiekowa_dzien_tygodnia_2017-2025.xlsx",
    "zgony": "source_data/zamachy_samobojcze_zakonczone_ZGONEM_grupa_wiekowa_dzien_tygodnia_2017-2025.xlsx",
    "detale": "source_data/zamachy_samobojcze_zrodlo_utrzymania_stan_swiadomosci_stan_zdrowia_kontakt_z_2017-2025.xlsx",
    "edukacja": "source_data/zamachy_samobojcze_stan_cywilny_wyksztalcenie_info_o_praca_nauka_2017-2025.xlsx"
}

# --- SŁOWNIK MAPUJĄCY KWP NA WOJEWÓDZTWA ---
kwp_to_wojewodztwo = {
    'KWP Bydgoszcz': 'Kujawsko-Pomorskie',
    'KWP Białystok': 'Podlaskie',
    'KWP Gdańsk': 'Pomorskie',
    'KWP Gorzów Wlkp.': 'Lubuskie',
    'KWP Katowice': 'Śląskie',
    'KWP Kielce': 'Świętokrzyskie',
    'KWP Kraków': 'Małopolskie',
    'KWP Łódź': 'Łódzkie',
    'KWP Lublin': 'Lubelskie',
    'KWP Olsztyn': 'Warmińsko-Mazurskie',
    'KWP Opole': 'Opolskie',
    'KWP Poznań': 'Wielkopolskie',
    'KWP Radom': 'Mazowieckie',
    'KWP Rzeszów': 'Podkarpackie',
    'KWP Szczecin': 'Zachodniopomorskie',
    'KSP Warszawa': 'Mazowieckie',
    'KWP Wrocław': 'Dolnośląskie'
}

def clean_load(path):
    """Wczytuje Excel i czyści nazwy kolumn."""
    if not os.path.exists(path):
        print(f"❌ Nie znaleziono pliku: {path}")
        return None
    print(f"Wczytywanie: {path}...")
    df = pd.read_excel(path)
    # Standaryzacja nazw (usuwanie enterów, podwójnych spacji)
    df.columns = [" ".join(str(c).replace('\n', ' ').split()).strip() for c in df.columns]
    if 'KWP' in df.columns: df['KWP'] = df['KWP'].astype(str).str.strip()
    if 'Rok' in df.columns:
        df['Rok'] = pd.to_numeric(df['Rok'], errors='coerce')
        df = df[df['Rok'].notna() & ~df['KWP'].str.contains('Polska', case=False, na=False)]
        df['Rok'] = df['Rok'].astype(int)
    return df

def find_cols(df, keywords):
    """Znajduje unikalne nazwy kolumn zawierające słowa kluczowe."""
    return list(set([c for c in df.columns if any(k.lower() in c.lower() for k in keywords)]))

def create_features_eda(files_dict):
    df_age = clean_load(files_dict["wiek_dni"])
    df_deaths = clean_load(files_dict["zgony"])
    df_details = clean_load(files_dict["detale"])
    df_edu = clean_load(files_dict["edukacja"])

    if any(d is None for d in [df_age, df_deaths, df_details, df_edu]): return None

    # 1. Znajdowanie kolumn bazowych
    C_TOTAL = [c for c in df_age.columns if "ogółem (PRÓBY I ZAKOŃCZONE ZGONEM)" in c][0]
    C_DEATHS = [c for c in df_deaths.columns if "zakończonych zgonem" in c][0]
    C_MALE = find_cols(df_details, ["mężczyzn"])[0]
    C_FEMALE = find_cols(df_details, ["kobiet"])[0]
    C_SOBER = find_cols(df_details, ["Trzeźwy"])[0]

    # 2. Kategorie wykształcenia
    edu_low = find_cols(df_edu, ["Podstawowe", "Gimnazjalne"])
    edu_mid = find_cols(df_edu, ["Zasadnicze zawodowe", "Średnie", "Policealne"])
    edu_high = find_cols(df_edu, ["Wyższe"])
    edu_unkn = find_cols(df_edu, ["Wykształcenie-Brak danych", "nieustalone"])
    all_edu_raw = list(set(edu_low + edu_mid + edu_high + edu_unkn))

    # 3. Łączenie danych (Merge)
    df = pd.merge(df_age, df_details, on=['Rok', 'KWP', C_TOTAL])
    df = pd.merge(df, df_deaths[['Rok', 'KWP', C_DEATHS]], on=['Rok', 'KWP'])
    df = pd.merge(df, df_edu[['Rok', 'KWP'] + all_edu_raw], on=['Rok', 'KWP'])

    # 4. Agregacja Mazowsza
    df['Województwo'] = df['KWP'].map(kwp_to_wojewodztwo)
    
    # Lista wszystkich surowych kolumn do zsumowania
    raw_numeric = list(set([C_TOTAL, C_DEATHS, C_MALE, C_FEMALE, C_SOBER] + all_edu_raw + \
                  [c for c in df.columns if any(k in c for k in ['Grupa', 'Dzień', 'Źródło', 'Stan', 'Informac', 'Kontakt'])]))
    
    for col in raw_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # SUMUJEMY LICZBY DLA WOJEWÓDZTW (Warszawa + Radom = jeden wiersz)
    df_agg = df.groupby(['Rok', 'Województwo'])[raw_numeric].sum().reset_index()
    df_agg['KWP'] = df_agg['Województwo']
    
    total = df_agg[C_TOTAL]

    # 5. OBLICZENIA WSKAŹNIKÓW (Na zagregowanych sumach)
    res = pd.DataFrame()
    res['Rok'] = df_agg['Rok']; res['KWP'] = df_agg['KWP']; res['Województwo'] = df_agg['Województwo']
    
    res['target_mortality_rate'] = (df_agg[C_DEATHS] / total).fillna(0)
    res['male_pct'] = (df_agg[C_MALE] / total).fillna(0)
    res['female_pct'] = (df_agg[C_FEMALE] / total).fillna(0)
    res['sober_pct'] = (df_agg[C_SOBER] / total).fillna(0)
    
    # Edukacja (Agregaty)
    res['edu_low_pct'] = df_agg[edu_low].sum(axis=1) / total
    res['edu_mid_pct'] = df_agg[edu_mid].sum(axis=1) / total
    res['edu_higher_pct'] = df_agg[edu_high].sum(axis=1) / total
    res['edu_unknown_pct'] = df_agg[edu_unkn].sum(axis=1) / total
    
    # Wiek i Cechy Suicydologiczne
    def sum_k(dataframe, keywords):
        cols = [c for c in dataframe.columns if any(k in c for k in keywords)]
        return dataframe[cols].sum(axis=1)

    res['youth_pct'] = sum_k(df_agg, ["'0-6'", "'7-12'", "'13-18'"]) / total
    res['young_adult_pct'] = sum_k(df_agg, ["'19-24'", "'25-29'", "'30-34'"]) / total
    res['middle_age_pct'] = sum_k(df_agg, ["'35-39'", "'40-44'", "'45-49'", "'50-54'", "'55-59'"]) / total
    res['senior_pct'] = sum_k(df_agg, ["'60-64'", "'65-69'", "'70-74'", "'75-79'", "'80-84'", "'85+'"]) / total
    res['substances_pct'] = sum_k(df_agg, ["alkoholu", "odurzających", "leków", "dopalaczy"]) / total
    
    stable = sum_k(df_agg, ["Praca", "Emerytura", "Renta"])
    unstable = sum_k(df_agg, ["Nie ma stałego źródła", "Zasiłek"])
    res['SES_instability_index'] = (unstable - stable) / total
    res['weekend_pct'] = sum_k(df_agg, ["sobota", "niedziela"]) / total
    
    return res.round(4)

# --- URUCHOMIENIE ---
try:
    df_final = create_features_eda(paths)
    if df_final is not None:
        df_final.to_csv("final_suicide_features_with_eda.csv", index=False)
        with open('dane_eda.pkl', 'wb') as f:
            pickle.dump(df_final, f)
        print("✅ Sukces! Wygenerowano plik 'dane_eda.pkl' z nowymi kategoriami edukacji.")
        print(df_final.head())
        print(df_final.columns)
except Exception as e:
    print(f"❌ Błąd krytyczny: {e}")