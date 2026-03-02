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

def create_features(files_dict):
    df_age = clean_load(files_dict["wiek_dni"])
    df_deaths = clean_load(files_dict["zgony"])
    df_details = clean_load(files_dict["detale"])
    df_edu = clean_load(files_dict["edukacja"])

    if any(d is None for d in [df_age, df_deaths, df_details, df_edu]):
        return None

    # Dynamiczne znajdowanie głównych kolumn
    C_TOTAL = find_col(df_age, "ogółem (PRÓBY I ZAKOŃCZONE ZGONEM)")
    C_DEATHS = find_col(df_deaths, "zakończonych zgonem")
    C_HIGHER_EDU = find_col(df_edu, "Wykształcenie - Wyższe")
    C_MALE = find_col(df_details, "W tym mężczyzn")

    # Łączenie
    df = pd.merge(df_age, df_details, on=['Rok', 'KWP', C_TOTAL])
    df = pd.merge(df, df_deaths[['Rok', 'KWP', C_DEATHS]], on=['Rok', 'KWP'])
    df = pd.merge(df, df_edu[['Rok', 'KWP', C_HIGHER_EDU]], on=['Rok', 'KWP'])

    # Konwersja na liczby wszystkich kolumn zawierających dane
    cols_to_convert = [C_TOTAL, C_DEATHS, C_HIGHER_EDU, C_MALE] + \
                      [c for c in df.columns if any(k in c for k in ['Grupa', 'Dzień', 'Źródło', 'Stan', 'Informac', 'Kontakt'])]
    
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- AGREGACJA DO POZIOMU WOJEWÓDZTWA ---
    # Dodajemy województwo do surowych danych
    df['Województwo'] = df['KWP'].map(kwp_to_wojewodztwo)
    
    # Wybieramy kolumny numeryczne do zsumowania (wszystkie dane surowe)
    numeric_cols = [c for c in cols_to_convert if c in df.columns]
    
    # Grupowanie po Roku i Województwie - to łączy KSP Warszawa i KWP Radom w jeden rekord
    df = df.groupby(['Rok', 'Województwo'])[numeric_cols].sum().reset_index()
    
    # Przypisujemy nazwę województwa do KWP, aby zachować spójność z final_cols
    df['KWP'] = df['Województwo']

    # Definiujemy total na zagregowanych danych
    total = df[C_TOTAL]

    # --- OBLICZENIA (FEATURE ENGINEERING) ---
    df['target_mortality_rate'] = (df[C_DEATHS] / total).fillna(0)
    df['male_pct'] = (df[C_MALE] / total).fillna(0)
    df['education_higher_pct'] = (df[C_HIGHER_EDU] / total).fillna(0)

    # Wiek - Agregacja
    def sum_cols(dataframe, keywords):
        cols = [dataframe.columns[i] for i, c in enumerate(dataframe.columns) if any(k in c for k in keywords)]
        return dataframe[cols].sum(axis=1)

    df['youth_pct'] = sum_cols(df, ["'0-6'", "'7-12'", "'13-18'"]) / total
    df['young_adult_pct'] = sum_cols(df, ["'19-24'", "'25-29'", "'30-34'"]) / total
    df['middle_age_pct'] = sum_cols(df, ["'35-39'", "'40-44'", "'45-49'", "'50-54'", "'55-59'"]) / total
    df['senior_pct'] = sum_cols(df, ["'60-64'", "'65-69'", "'70-74'", "'75-79'", "'80-84'", "'85+'"]) / total

    # Stabilność dochodów (SES)
    stable_income = sum_cols(df, ["Źródło utrzymania - Praca", "Źródło utrzymania - Emerytura", "Źródło utrzymania - Renta"])
    unstable_income = sum_cols(df, ["Nie ma stałego źródła", "Zasiłek"])
    df['SES_instability_index'] = (unstable_income - stable_income) / total

    # Stan świadomości
    df['substances_pct'] = sum_cols(df, ["alkoholu", "środków odurzających", "leków", "dopalaczy"]) / total

    # Kontakt z instytucjami
    inst_cols = [c for c in df.columns if 'Kontakt z instytucjami' in c and 'Brak możliwości' not in c]
    df['institution_contact_pct'] = df[inst_cols].sum(axis=1) / total

    # Czas
    df['weekend_pct'] = sum_cols(df, ["sobota", "niedziela"]) / total

    # Selekcja końcowa (KWP teraz zawiera nazwę województwa)
    final_cols = [
        'Rok', 'KWP', 'Województwo', 'target_mortality_rate', 'male_pct', 'education_higher_pct',
        'youth_pct', 'young_adult_pct', 'middle_age_pct', 'senior_pct',
        'SES_instability_index', 'substances_pct', 'institution_contact_pct', 'weekend_pct'
    ]
    
    return df[final_cols].round(4)



# --- URUCHOMIENIE I ZAPIS ---
try:
    df_final = create_features(paths)
    if df_final is not None:
        # 1. Zapis do CSV
        df_final.to_csv("final_suicide_features.csv", index=False)
        print("✅ Sukces! Plik 'final_suicide_features.csv' został zapisany.")
        
        # 2. Zapis do Pickle
        with open('dane.pkl', 'wb') as f:
            pickle.dump(df_final, f)
        print("✅ Dane zapisane pomyślnie do 'dane.pkl'!")
        
        print("\nPodgląd danych:")
        print(df_final.head())
    else:
        print("❌ Nie udało się wygenerować cech (sprawdź pliki wejściowe).")

except Exception as e:
    print(f"❌ Wystąpił niespodziewany błąd: {e}")

