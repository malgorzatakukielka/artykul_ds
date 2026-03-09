import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
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

# 3. Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 4. Wyniki
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Ewaluacja na zbiorze testowym
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters found:", best_params)
print(f"MAE on Test Set: {mae:.4f}")
print(f"R2 Score on Test Set: {r2:.4f}")

# Definicja modelu z najlepszymi parametrami
final_model = RandomForestRegressor(
    max_depth=8, 
    max_features='sqrt', 
    min_samples_leaf=1, 
    min_samples_split=5, 
    n_estimators=100, 
    random_state=42
)

final_model.fit(X_train, y_train)

# Zapisanie ostatecznego modelu
with open('final_optimized_suicide_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)