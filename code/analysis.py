import pickle
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn
import geodatasets
#ładowanie danych
try:
    with open('dane.pkl', 'rb') as f:
        df = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")

# Analiza Trendów Czasowych
# --- KONFIGURACJA ---
INPUT_FILE = "final_suicide_features.csv"

def generate_maps():
    # 1. Wczytanie danych z Twojego pliku CSV
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Nie znaleziono pliku {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    df['Województwo'] = df['Województwo'].str.lower()

    # Wyznaczamy wspólny zakres dla wszystkich map
    global_min = df['target_mortality_rate'].min()
    global_max = df['target_mortality_rate'].max()
    print(f"Stała skala: {global_min:.2f} do {global_max:.2f}")

    # 2. Pobieranie mapy
    GEOJSON_URL = "https://github.com/ppatrzyk/polska-geojson/raw/master/wojewodztwa/wojewodztwa-max.geojson"
    print("Pobieranie mapy Polski (GeoJSON)...")
    try:
        poland_map = gpd.read_file(GEOJSON_URL)
        poland_map['nazwa'] = poland_map['nazwa'].str.lower()
    except Exception as e:
        print(f"❌ Błąd mapy: {e}")
        return
    
    # 3. Pętla generująca mapy dla każdego roku
    years = sorted(df['Rok'].unique())
    
    for year in years:
        print(f"Generowanie mapy za rok {year}...")
        
        # Filtrowanie i agregacja (Mazowsze ma 2 wpisy KWP, bierzemy średnią)
        df_year = df[df['Rok'] == year].groupby('Województwo')['target_mortality_rate'].mean().reset_index()
        
        # Łączenie danych tabelarycznych z mapą
        merged = poland_map.merge(df_year, left_on='nazwa', right_on='Województwo')
        
        # Tworzenie wykresu
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Rysowanie mapy
        merged.plot(
            column='target_mortality_rate', 
            cmap='OrRd', 
            linewidth=0.8, 
            ax=ax, 
            edgecolor='0.8',
            vmin = global_min,
            vmax = global_max,
            legend=True,
            legend_kwds={'label': "Wskaźnik śmiertelności (Zgony / Próby)", 'orientation': "horizontal", 'pad': 0.01}
        )
        
        # Dodanie etykiet z wartościami na mapie
        for idx, row in merged.iterrows():
            # Wyznaczanie środka województwa dla tekstu
            centroid = row['geometry'].centroid
            ax.annotate(
                text=f"{row['target_mortality_rate']:.2f}", 
                xy=(centroid.x, centroid.y),
                horizontalalignment='center', 
                fontsize=9, 
                color='black',
                weight='bold'
            )

        ax.set_title(f'Śmiertelność zamachów samobójczych w Polsce - Rok {year}', fontsize=15)
        ax.axis('off') # Ukrycie współrzędnych geograficznych
        
        # Zapisywanie pliku
        plt.savefig(f"mapy_smiertelnosci/mapa_smiertelnosci_{year}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("✅ Wszystkie mapy zostały wygenerowane i zapisane jako pliki .png")

if __name__ == "__main__":
    generate_maps()

