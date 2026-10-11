import joblib
import numpy as np

# 1. Завантажуємо словник з файлу
saved_data = joblib.load('ews_model.joblib')

# 2. Витягуємо потрібні компоненти
model = saved_data['model']
features = saved_data['features']
threshold = saved_data.get('threshold', 'не вказано')
horizon = saved_data.get('horizon_s', 'не вказано')

print("=== Інформація про модель ===")
print(f"Горизонт прогнозу: {horizon} секунд")
print(f"Поріг спрацьовування: {threshold}")
print(f"Кількість потрібних параметрів: {len(features)}")
print(f"Список очікуваних параметрів: {features}")
print("=============================\n")

# 3. Створюємо тестові дані
# Щоб не було помилок, створюємо масив з нулів, розмір якого точно 
# відповідає кількості параметрів, які очікує модель.
new_data = np.zeros((1, len(features)))

# 4. Робимо прогноз
prediction = model.predict(new_data)
print(f"Результат прогнозу для тестових даних (всі нулі): {prediction[0]}")