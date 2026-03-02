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
    """Wczytuje Excel i czyści nazwy kolumn ze wszystkich białych znaków."""
    if not os.path.exists(path):
        print(f"❌ Nie znaleziono pliku: {path}")
        return None
    
    print(f"Wczytywanie: {path}...")
    df = pd.read_excel(path)
    
    # Czyszczenie nazw kolumn: zamiana \n na spację, usuwanie podwójnych spacji i strip
    df.columns = [
        " ".join(str(c).replace('\n', ' ').split()).strip() 
        for c in df.columns
    ]
    
    # Standaryzacja kluczowych kolumn
    if 'KWP' in df.columns:
        df['KWP'] = df['KWP'].astype(str).str.strip()
    
    if 'Rok' in df.columns:
        df['Rok'] = pd.to_numeric(df['Rok'], errors='coerce')
        # Usuwamy wiersze bez roku i sumaryczne dla Polski
        df = df[df['Rok'].notna()]
        df = df[~df['KWP'].str.contains('Polska', case=False, na=False)]
        df['Rok'] = df['Rok'].astype(int)
    
    return df

def find_col(df, keyword):
    """Pomocnicza funkcja do znajdowania kolumny po fragmencie nazwy."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

def create_features_eda(files_dict):
    df_age = clean_load(files_dict["wiek_dni"])
    df_deaths = clean_load(files_dict["zgony"])
    df_details = clean_load(files_dict["detale"])
    df_edu = clean_load(files_dict["edukacja"])

    if any(d is None for d in [df_age, df_deaths, df_details, df_edu]):
        return None

    # Dynamiczne znajdowanie głównych kolumn
# Znajdowanie kolumn bazowych (surowe liczby)
    C_TOTAL = find_col(df_age, "ogółem (PRÓBY I ZAKOŃCZONE ZGONEM)")
    C_DEATHS = find_col(df_deaths, "zakończonych zgonem")
    C_HIGHER_EDU = find_col(df_edu, "Wykształcenie - Wyższe")
    C_MALE = find_col(df_details, "W tym mężczyzn")
    C_FEMALE = find_col(df_details, "W tym kobiet")
    C_SOBER = find_col(df_details, "Trzeźwy(a)")
    C_SINGLE = find_col(df_edu, "Kawaler/panna")
    C_UNEMPLOYED = find_col(df_edu, "Bezrobotny")


    # Łączenie
    df = pd.merge(df_age, df_details, on=['Rok', 'KWP', C_TOTAL])
    df = pd.merge(df, df_deaths[['Rok', 'KWP', C_DEATHS]], on=['Rok', 'KWP'])
    df = pd.merge(df, df_edu[['Rok', 'KWP', C_HIGHER_EDU, C_SINGLE, C_UNEMPLOYED]], on=['Rok', 'KWP'])

    eda_raw_cols = [C_TOTAL, C_DEATHS, C_HIGHER_EDU, C_MALE, C_FEMALE, C_SOBER, C_SINGLE, C_UNEMPLOYED]
    other_data_cols = [c for c in df.columns if any(k in c for k in ['Grupa', 'Dzień', 'Źródło', 'Stan', 'Informac', 'Kontakt'])]
    
    cols_to_convert = list(set(eda_raw_cols + other_data_cols)) # set usuwa duplikaty
    
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- AGREGACJA DO POZIOMU WOJEWÓDZTWA ---
    # Dodajemy województwo do surowych danych
    df['Województwo'] = df['KWP'].map(kwp_to_wojewodztwo)
    
    # Wybieramy kolumny numeryczne do zsumowania (wszystkie dane surowe)
    numeric_cols = [c for c in cols_to_convert if c in df.columns]
    
    # Grupowanie po Roku i Województwie
    df = df.groupby(['Rok', 'Województwo'])[numeric_cols].sum().reset_index()
    
    # Przypisujemy nazwę województwa do KWP, aby zachować spójność z final_cols
    df['KWP'] = df['Województwo']

    # Definiujemy total na zagregowanych danych
    total = df[C_TOTAL]

    # --- OBLICZENIA (Wskaźniki EDA) ---
    df['target_mortality_rate'] = (df[C_DEATHS] / total).fillna(0)
    df['male_pct'] = (df[C_MALE] / total).fillna(0)
    df['female_pct'] = (df[C_FEMALE] / total).fillna(0)
    df['education_higher_pct'] = (df[C_HIGHER_EDU] / total).fillna(0)
    df['single_pct'] = (df[C_SINGLE] / total).fillna(0)
    df['unemployed_pct'] = (df[C_UNEMPLOYED] / total).fillna(0)
    df['sober_pct'] = (df[C_SOBER] / total).fillna(0)

    # Wiek - Agregacja
    def sum_keywords(dataframe, keywords):
        cols = [dataframe.columns[i] for i, c in enumerate(dataframe.columns) if any(k in c for k in keywords)]
        return dataframe[cols].sum(axis=1)

    df['youth_pct'] = sum_keywords(df, ["'0-6'", "'7-12'", "'13-18'"]) / total
    df['young_adult_pct'] = sum_keywords(df, ["'19-24'", "'25-29'", "'30-34'"]) / total
    df['middle_age_pct'] = sum_keywords(df, ["'35-39'", "'40-44'", "'45-49'", "'50-54'", "'55-59'"]) / total
    df['senior_pct'] = sum_keywords(df, ["'60-64'", "'65-69'", "'70-74'", "'75-79'", "'80-84'", "'85+'"]) / total

    # Stabilność dochodów (SES)
    stable = sum_keywords(df, ["Praca", "Emerytura", "Renta"])
    unstable = sum_keywords(df, ["Nie ma stałego źródła", "Zasiłek"])
    df['SES_instability_index'] = (unstable - stable) / total
    df['substances_pct'] = sum_keywords(df, ["alkoholu", "odurzających", "leków", "dopalaczy"]) / total

# Instytucje i Czas
    inst_cols = [c for c in df.columns if 'Kontakt z instytucjami' in c and 'Brak możliwości' not in c]
    df['institution_contact_pct'] = df[inst_cols].sum(axis=1) / total
    df['weekend_pct'] = sum_keywords(df, ["sobota", "niedziela"]) / total

    # Czas
    df['weekend_pct'] = sum_keywords(df, ["sobota", "niedziela"]) / total

# Selekcja końcowa (rozszerzona o EDA)
    final_cols = [
        'Rok', 'KWP', 'Województwo', 'target_mortality_rate', 
        'male_pct', 'female_pct', 'youth_pct', 'young_adult_pct', 
        'middle_age_pct', 'senior_pct', 'education_higher_pct', 
        'single_pct', 'unemployed_pct', 'SES_instability_index', 
        'substances_pct', 'sober_pct', 'institution_contact_pct', 'weekend_pct'
    ]
    
    return df[final_cols].round(4)



# --- URUCHOMIENIE I ZAPIS ---
try:
    df_eda = create_features_eda(paths)
    if df_eda is not None:
        df_eda.to_csv("final_suicide_features_with_eda.csv", index=False)
        with open('dane_eda.pkl', 'wb') as f:
            pickle.dump(df_eda, f)
        print("✅ Sukces! Plik 'dane_eda.pkl' został zapisany.")
        print(df_eda.head())
except Exception as e:
    print(f"❌ Błąd: {e}")

