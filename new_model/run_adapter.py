



import joblib
import numpy as np
from datasets import load_dataset

# 1. Завантажуємо вашу модель та список зі 117 колонок
saved_data = joblib.load('ews_model.joblib')
model = saved_data['model']
expected_features = saved_data['features']
threshold = saved_data.get('threshold', 0.5)

print(f"Модель завантажена. Очікується {len(expected_features)} колонок.")

# 2. Завантажуємо ОДИН експеримент з даних хакатону через Hugging Face
print("Завантаження даних хакатону (це може зайняти кілька секунд)...")
dataset = load_dataset("Sophelio/fusion-equilibrium-challenge", "diii_d_train", split="train", streaming=True)
hackathon_shot = next(iter(dataset)) # Беремо перший доступний shot

# 3. Функція-адаптер: перетворює "сирі" масиви хакатону на 117 скалярних ознак
def create_117_features(shot_data, feature_names):
    # Створюємо словник, де всі 117 колонок спочатку заповнені нулями
    features = {feat: 0.0 for feat in feature_names}
    
    # --- ВИЛУЧАЄМО РЕАЛЬНІ ДАНІ, ЯКІ Є В ХАКАТОНІ ---
    
    # 1. Струм плазми (ip_MA)
    if "magnetics_plasma_current" in shot_data:
        # Беремо середнє значення масиву. В хакатоні це кілоампери (kA), моделі треба мегаампери (MA)
        ip_ka = np.nanmean(shot_data["magnetics_plasma_current"])
        features["ip_MA"] = ip_ka / 1000.0
        
    # 2. Температура електронів (te0_keV)
    if "thomson_core_Te" in shot_data:
        # Дані Томпсона - це матриця (час х канали). Беремо максимум (як центр плазми).
        # В хакатоні це електрон-вольти (eV), моделі треба кілоелектрон-вольти (keV)
        te_ev = np.nanmax(shot_data["thomson_core_Te"])
        features["te0_keV"] = te_ev / 1000.0
        
    # 3. Тривалість пострілу (duration_s)
    if "efit_times" in shot_data:
        # efit_times в мілісекундах, переводимо в секунди
        features["duration_s"] = shot_data["efit_times"][-1] / 1000.0
        
    # --- ТУТ МОЖНА ДОДАТИ ІНШІ РОЗРАХУНКИ ---
    # Наприклад, можна розрахувати нахил струму для 'ip_MA__slope100ms' і т.д.
    
    # Перетворюємо заповнений словник назад у масив для моделі, 
    # строго дотримуючись порядку з expected_features
    row_array = np.array([[features[feat] for feat in feature_names]])
    return row_array

# 4. Формуємо масив на 117 колонок
X_test = create_117_features(hackathon_shot, expected_features)

print("\n=== Згенеровані дані для моделі (фрагмент) ===")
# Виведемо перші 10 параметрів, щоб перевірити, що не всюди нулі
for feat, val in zip(expected_features[:10], X_test[0][:10]):
    print(f"{feat}: {val:.4f}")

# 5. Робимо прогноз
print("\n=== Результат прогнозу ===")
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(X_test)[0][1]
    is_alarm = 1 if prob >= threshold else 0
    print(f"Ймовірність зриву: {prob:.4f} (Поріг: {threshold:.4f})")
    print(f"Вердикт: {'ТРИВОГА (1)' if is_alarm else 'НОРМА (0)'}")
else:
    pred = model.predict(X_test)[0]
    print(f"Вердикт моделі: {pred}")