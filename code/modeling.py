import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

try:
    with open('dane.pkl', 'rb') as f:
        df = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")

# Dodanie zmiennej is_covid
df['is_covid'] = df['Rok'].isin([2020, 2021]).astype(int)

# Cechy X
features = [
    'male_pct',             # Płeć
    'youth_pct',            # Udział młodzieży
    'senior_pct',           # Udział seniorów
    'substances_pct',       # Świadomość
    'SES_instability_index',# Ekonomia
    'is_covid',             # Pandemia
    'weekend_pct',          # Czas
    'education_higher_pct'  # Wykształcenie
]
# Cecha Y (target)
target = 'target_mortality_rate'

# Train/Test split
train = df[df['Rok'] < 2024]
test = df[df['Rok'] >= 2024]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

#do przeniesienia do analiz, kiedyś to zrobię ig
# sns.heatmap(df[features + [target]].corr(), annot= True, fmt= ".2f")
# plt.show()


# Trenowanie modelu Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Ewaluacja modelu
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- WYNIKI MODELU ---")
print(f"Błąd średni (MAE): {mae:.4f}")
print(f"Wyjaśniona wariancja (R2): {r2:.4f}")

# WAŻNOŚĆ CECH
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
print(importances)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='teal')
plt.title('Ranking czynników wpływających na śmiertelność zamachów', fontsize=14)
plt.xlabel('Poziom istotności (Feature Importance)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_feature_importance.png', dpi=300)

# 7. Zapisanie modelu
with open('suicide_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model został wytrenowany. Wykres ważności cech zapisano jako 'model_feature_importance.png'.")


# Porównanie wartości rzeczywistych i przewidzianych (dla zbioru testowego)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.title(f'Model Random Forest: Rzeczywistość vs Przewidywania\n(R2 = {r2:.2f})', fontsize=14)
plt.xlabel('Rzeczywista śmiertelność (Zgony/Próby)', fontsize=12)
plt.ylabel('Przewidziana śmiertelność przez model', fontsize=12)
plt.tight_layout()
plt.savefig('model_actual_vs_predicted.png', dpi=300)