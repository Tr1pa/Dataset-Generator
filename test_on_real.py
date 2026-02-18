import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "runs", "metro_damage", "weights", "best.pt")
image_path = os.path.join(BASE_DIR, "image.png")

if not os.path.exists(image_path):
    print(f"ОШИБКА: Файл не найден по пути: {image_path}")
    print("Пожалуйста, положите картинку test_real.jpg в папку со скриптом.")
else:
    model = YOLO(model_path)

    results = model.predict(source=image_path, save=True, conf=0.25)

    print(f"\nПроверка завершена!")
    print(f"Результат сохранен в папку: {results[0].save_dir}")