




import argparse
import joblib
import pandas as pd
import numpy as np
import os

def calculate_dynamics(group):
    roll_cols = ['diag_d_alpha', 'diag_soft_xray', 'diag_locked_mode', 'diag_neutron_rate', 'diag_v_loop_V']
    slope_cols = ['f_greenwald', 'beta_n', 'q95', 'li', 'te0_keV', 'w_th_MJ', 'p_rad_MW']
    
    windows = {'100ms': 5, '500ms': 25}
    new_cols = {}
    
    for w_name, w_size in windows.items():
        for col in roll_cols:
            if col in group.columns:
                new_cols[f'{col}__mean{w_name}'] = group[col].rolling(window=w_size, min_periods=1).mean()
                new_cols[f'{col}__max{w_name}'] = group[col].rolling(window=w_size, min_periods=1).max()
                new_cols[f'{col}__std{w_name}'] = group[col].rolling(window=w_size, min_periods=1).std().fillna(0)
                
        dt = 0.1 if w_name == '100ms' else 0.5
        for col in slope_cols:
            if col in group.columns:
                slope_val = (group[col] - group[col].shift(w_size)) / dt
                new_cols[f'{col}__slope{w_name}'] = slope_val.fillna(0)
                
    if not new_cols:
        return group
        
    new_df = pd.DataFrame(new_cols, index=group.index)
    return pd.concat([group, new_df], axis=1)

def main():
    # Налаштування аргументів командного рядка
    parser = argparse.ArgumentParser(description="Утиліта для обробки CSV файлів моделлю EWS")
    parser.add_argument("input_csv", help="Шлях до вхідного CSV файлу")
    parser.add_argument("--model", default="ews_model.joblib", help="Шлях до моделі (за замовчуванням ews_model.joblib)")
    parser.add_argument("--output", help="Шлях до вихідного файлу для збереження результатів (опціонально)")
    
    args = parser.parse_args()

    # Перевірка наявності файлів
    if not os.path.exists(args.input_csv):
        print(f"❌ Помилка: Вхідний файл '{args.input_csv}' не знайдено!")
        return
        
    if not os.path.exists(args.model):
        print(f"❌ Помилка: Файл моделі '{args.model}' не знайдено!")
        return

    # 1. Завантаження моделі
    print(f"[*] Завантаження моделі '{args.model}'...")
    saved_data = joblib.load(args.model)
    model = saved_data['model']
    expected_features = saved_data['features']
    threshold = saved_data.get('threshold', 0.057)

    # 2. Завантаження даних
    print(f"[*] Читання файлу '{args.input_csv}'...")
    df = pd.read_csv(args.input_csv)
    
    # 3. Обчислення динаміки
    print("[*] Обчислення часової динаміки (100ms та 500ms)... Зачекайте.")
    df_original = df.copy()
    
    if 'shot_id' in df.columns:
        df = df.groupby('shot_id', group_keys=False).apply(calculate_dynamics)
        df['shot_id'] = df_original['shot_id']
    else:
        # Якщо немає колонки shot_id, обробляємо весь файл як один експеримент
        df = calculate_dynamics(df)

    # 4. Підготовка 117 колонок (якщо якихось колонок немає у файлі, заповнюємо їх нулями)
    print("[*] Підготовка фінальної таблиці...")
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0.0
            
    X = df[expected_features].fillna(0)

    print(f"[*] Сформовано вектор ознак для моделі: {X.shape[1]} колонок.")
    
    # 5. Прогноз
    print("[*] Генерація прогнозів...")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = model.predict(X)

    df['predicted_risk'] = probabilities
    df['alarm_triggered'] = (probabilities >= threshold).astype(int)

    # 6. Вивід результатів
    alarms = df['alarm_triggered'].sum()
    print("\n" + "="*30)
    print("      РЕЗУЛЬТАТИ АНАЛІЗУ")
    print("="*30)
    print(f"Оброблено рядків: {len(df)}")
    print(f"Кількість тривог (ризик >= {threshold}): {alarms}")

    if 'shot_id' not in df.columns:
        df = df.reset_index()
    display_cols = [c for c in ['shot_id', 'time_s', 'predicted_risk'] if c in df.columns]
    
    top_risks = df.sort_values('predicted_risk', ascending=False)[display_cols].head(5)
    print("\nТоп-5 моментів з найвищим ризиком зриву:")
    print(top_risks.to_string(index=False))

    # 7. Збереження у новий файл
    output_file = args.output
    if not output_file:
        # Створюємо ім'я автоматично, якщо не вказано (наприклад: timeseries_predictions.csv)
        base_name = os.path.splitext(args.input_csv)[0]
        output_file = f"{base_name}_predictions.csv"
        
    df.to_csv(output_file, index=False)
    print(f"\n Усі результати успішно збережено у файл: {output_file}")

if __name__ == '__main__':
    main()