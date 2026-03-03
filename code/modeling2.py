import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. Wczytanie danych
try:
    with open('dane.pkl', 'rb') as f:
        df = pickle.load(f)
    print("Dane wczytane! Masz dostęp do zmiennej 'df'.")
except FileNotFoundError:
    print("Błąd: Plik 'dane.pkl' nie istnieje. Uruchom najpierw pobieranie.")

# 2. Wybór TYLKO istotnych cech (Top 4)
features_reduced = [
    'senior_pct',      # Ważność: ~51%
    'male_pct',        # Ważność: ~24%
    'substances_pct',  # Ważność: ~14%
    'youth_pct'        # Ważność: ~5%
]
target = 'target_mortality_rate'

# 3. Podział chronologiczny
train_data = df[df['Rok'] < 2024]
test_data = df[df['Rok'] >= 2024]

X_train, y_train = train_data[features_reduced], train_data[target]
X_test, y_test = test_data[features_reduced], test_data[target]

# 4. Trenowanie uproszczonego modelu
# Zwiększamy liczbę drzew (n_estimators), aby model był jeszcze stabilniejszy
model_slim = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
model_slim.fit(X_train, y_train)

# 5. Ewaluacja
y_pred = model_slim.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- WYNIKI UPROSZCZONEGO MODELU ---")
print(f"Błąd średni (MAE): {mae:.4f}")
print(f"Wyjaśniona wariancja (R2): {r2:.4f}")

# 6. Ważność cech w nowym modelu
importances = pd.Series(model_slim.feature_importances_, index=features_reduced).sort_values(ascending=True)
print(importances)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='darkred')
plt.title('Zredukowany model: Główne czynniki śmiertelności', fontsize=14)
plt.xlabel('Waga (Importance)')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('model_slim_importance.png', dpi=300)

# 7. Wykres Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkred', label='Przewidywania modelu')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Idealne dopasowanie')
plt.title(f'Rzeczywistość vs Model (Uproszczony)\nR2 = {r2:.2f}', fontsize=14)
plt.xlabel('Rzeczywista śmiertelność')
plt.ylabel('Przewidziana śmiertelność')
plt.legend()
plt.savefig('model_slim_actual_vs_predicted.png', dpi=300)

# 8. Zapisanie nowego modelu
with open('suicide_model_slim.pkl', 'wb') as f:
    pickle.dump(model_slim, f)

print("\n✅ Uproszczony model gotowy. Sprawdź, czy R2 utrzymał się na podobnym poziomie.")