import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset_yolo", "data.yaml")
PROJECT = os.path.join(BASE_DIR, "runs")


def step1_train():
    """Обучение модели."""
    from ultralytics import YOLO

    print("=" * 55)
    print("  ШАГ 1: ОБУЧЕНИЕ")
    print("=" * 55)

    model = YOLO("yolov8n.pt")  # скачает автоматически

    model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=16,
        name="metro_damage",
        project=PROJECT,
        patience=10,       # ранняя остановка если нет улучшений
        save=True,
        plots=True,
    )

    print("\n  Модель сохранена в:")
    print(f"  {PROJECT}/metro_damage/weights/best.pt")


def step2_validate():
    """Валидация на val-сете."""
    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  ШАГ 2: ВАЛИДАЦИЯ")
    print("=" * 55)

    best = os.path.join(PROJECT, "metro_damage", "weights", "best.pt")
    if not os.path.exists(best):
        print(f"  Нет файла {best}")
        print("  Сначала запустите обучение")
        return

    model = YOLO(best)
    results = model.val(data=DATASET)

    print(f"\n  mAP50:    {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    print(f"  Precision: {results.box.mp:.3f}")
    print(f"  Recall:    {results.box.mr:.3f}")


def step3_test_images():
    """Тест на реальных фото из raw_images."""
    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  ШАГ 3: ТЕСТ НА РЕАЛЬНЫХ ФОТО")
    print("=" * 55)

    best = os.path.join(PROJECT, "metro_damage", "weights", "best.pt")
    raw = os.path.join(BASE_DIR, "raw_images")
    out = os.path.join(BASE_DIR, "test_predictions")

    if not os.path.exists(best):
        print("  Нет обученной модели")
        return

    model = YOLO(best)

    results = model.predict(
        source=raw,
        save=True,
        save_txt=True,
        project=out,
        name="results",
        conf=0.25,
        imgsz=640,
    )

    total = len(results)
    detected = sum(1 for r in results if len(r.boxes) > 0)

    print(f"\n  Обработано: {total}")
    print(f"  С детекциями: {detected}")
    print(f"  Результаты: {out}/results/")


def step4_export():
    """Экспорт модели для продакшена."""
    from ultralytics import YOLO

    print("\n" + "=" * 55)
    print("  ШАГ 4: ЭКСПОРТ")
    print("=" * 55)

    best = os.path.join(PROJECT, "metro_damage", "weights", "best.pt")
    if not os.path.exists(best):
        print("  Нет обученной модели")
        return

    model = YOLO(best)

    model.export(format="onnx", imgsz=640)
    print("  Экспортировано в ONNX")


def main():
    print("=" * 55)
    print("  ПАЙПЛАЙН ОБУЧЕНИЯ YOLO")
    print("=" * 55)
    print(f"""
  Что делаем:
    1. train    — обучение (~10-30 мин)
    2. val      — проверка метрик
    3. test     — тест на raw_images
    4. export   — экспорт в ONNX
    5. all      — всё по порядку

  Использование:
    python train.py train
    python train.py val
    python train.py test
    python train.py export
    python train.py all
""")

    if len(sys.argv) < 2:
        print("  Укажите команду!")
        print("  python train.py all")
        return

    cmd = sys.argv[1].lower()

    if cmd in ("train", "all"):
        step1_train()

    if cmd in ("val", "all"):
        step2_validate()

    if cmd in ("test", "all"):
        step3_test_images()

    if cmd in ("export", "all"):
        step4_export()

    if cmd == "all":
        print(f"\n{'='*55}")
        print("  ВСЁ ГОТОВО!")
        print(f"  Модель: {PROJECT}/metro_damage/weights/best.pt")
        print(f"  Тесты:  {BASE_DIR}/test_predictions/results/")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()  