import joblib
import pandas as pd
import numpy as np

# 1. Завантаження моделі
print("Завантаження моделі...")
saved_data = joblib.load('ews_model.joblib')
model = saved_data['model']
expected_features = saved_data['features']
threshold = saved_data.get('threshold', 0.057)

# 2. Завантаження даних
print("Читання timeseries.csv...")
df = pd.read_csv('timeseries.csv')

# 3. Оптимізована функція для розрахунку часової динаміки
def calculate_dynamics(group):
    roll_cols = ['diag_d_alpha', 'diag_soft_xray', 'diag_locked_mode', 'diag_neutron_rate', 'diag_v_loop_V']
    slope_cols = ['f_greenwald', 'beta_n', 'q95', 'li', 'te0_keV', 'w_th_MJ', 'p_rad_MW']
    
    windows = {'100ms': 5, '500ms': 25}
    
    # Створюємо словник для зберігання нових колонок (усуває фрагментацію пам'яті)
    new_cols = {}
    
    for w_name, w_size in windows.items():
        for col in roll_cols:
            new_cols[f'{col}__mean{w_name}'] = group[col].rolling(window=w_size, min_periods=1).mean()
            new_cols[f'{col}__max{w_name}'] = group[col].rolling(window=w_size, min_periods=1).max()
            new_cols[f'{col}__std{w_name}'] = group[col].rolling(window=w_size, min_periods=1).std().fillna(0)
            
        dt = 0.1 if w_name == '100ms' else 0.5
        for col in slope_cols:
            slope_val = (group[col] - group[col].shift(w_size)) / dt
            new_cols[f'{col}__slope{w_name}'] = slope_val.fillna(0)
            
    # Перетворюємо словник у DataFrame і приєднуємо до групи ЗА ОДИН РАЗ
    new_df = pd.DataFrame(new_cols, index=group.index)
    return pd.concat([group, new_df], axis=1)

# 4. Застосовуємо розрахунки
print("Розрахунок динаміки (100ms та 500ms)... Зачекайте кілька секунд.")
df_original = df.copy()
df = df.groupby('shot_id', group_keys=False).apply(calculate_dynamics)
df['shot_id'] = df_original['shot_id']  # Залізно повертаємо номер експерименту

# 5. Відбираємо строго 117 колонок
print("Підготовка фінальної таблиці...")
try:
    X = df[expected_features].fillna(0)
except KeyError as e:
    print(f"Помилка: не знайдено колонки {e}")
    exit()

# 6. Робимо прогноз
print("Генерація прогнозів...")
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
else:
    probabilities = model.predict(X)

df['predicted_risk'] = probabilities
df['alarm_triggered'] = (probabilities >= threshold).astype(int)

alarms = df['alarm_triggered'].sum()
print("\n=== РЕЗУЛЬТАТИ АНАЛІЗУ ===")
print(f"Оброблено рядків: {len(df)}")
print(f"Кількість тривог (моментів, коли ризик >= {threshold}): {alarms}")
# Якщо pandas сховав shot_id в індекс - витягуємо його назад
if 'shot_id' not in df.columns:
    df = df.reset_index()

# Перевіряємо, які колонки реально доступні для виводу
display_cols = [c for c in ['shot_id', 'time_s', 'predicted_risk'] if c in df.columns]

top_risks = df.sort_values('predicted_risk', ascending=False)[display_cols].head(5)
print("\nТоп-5 моментів з найвищим ризиком зриву:")
print(top_risks.to_string(index=False))