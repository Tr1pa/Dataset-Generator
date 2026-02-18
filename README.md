# Metro Damage AI Detection 🚇🤖

[Русское описание ниже](#russian-description)

This project provides a complete end-to-end pipeline for detecting interior damage in metro cars (damaged seats, dirty floors, and corroded metal). It solves the problem of data scarcity by using **Synthetic Data Generation** via state-of-the-art AI models.

## 🌟 Key Features
- **AI-Driven Synthesis:** Uses generative models (default: `gpt-5-image-mini` via OpenRouter) to create realistic training data.
- **Customizable:** You can swap the generation model in `generate_data.py` to `flux-1-schnell`, `stable-diffusion-3`, etc.
- **Full Pipeline:** From API-based generation and heavy augmentation to YOLOv8 training and ONNX export.
- **High Accuracy:** The model achieved an **mAP50-95 of 0.995** on the synthetic validation set.

## 🚀 Execution Order
To build the project from scratch, run the scripts in this specific order:

1.  **`generate_data.py`**: Connects to OpenRouter API to generate the initial set of images based on Moscow Metro car styles (Old, Mid-era, and Modern).
2.  **`augment.py`**: Applies various filters (noise, blur, rotation, color shifts) to the generated images to multiply the dataset size.
3.  **`data.py`**: Automatically splits the augmented data into `train` and `val` sets and creates the `data.yaml` file for YOLO.
4.  **`main.py`**: The primary orchestrator. Run `python main.py all` to perform training (50 epochs), validation, testing on raw images, and export to ONNX.
5.  **`test_on_real.py`**: Use this script to run the final `best.pt` model on any real-world photo (`image.png`).

## 📊 Results
- **Model:** YOLOv8 Nano (6.2 MB)
- **Precision:** 0.998 / **Recall:** 0.997
- **Inference:** ~60ms (CPU)

---

<a name="russian-description"></a>

# Детекция повреждений в метро (YOLOv8) 🚇🤖

Проект представляет собой полный цикл разработки системы компьютерного зрения для обнаружения дефектов внутри вагонов метро (порванные сиденья, грязный пол, ржавчина). Главная фишка — **автоматическая генерация датасета**.

## 🌟 Особенности
- **Синтетика на базе ИИ:** Создание реалистичных обучающих данных с помощью нейросетей через OpenRouter API.
- **Гибкость:** По умолчанию используется `gpt-5-image-mini`, но в `generate_data.py` можно выставить любую модель (Flux, SD3 и др.).
- **Промышленный стандарт:** Легкая модель YOLOv8n, экспортированная в **ONNX** для работы в реальном времени.
- **Высокие метрики:** mAP50-95 достигает **0.995** на валидационной выборке.

## 🚀 Порядок запуска
Для воспроизведения результата запускайте файлы строго в этой последовательности:

1.  **`generate_data.py`**: Генерация базовых изображений через API. Создает сцены с повреждениями в стилистике Московского метрополитена.
2.  **`augment.py`**: Программная аугментация (повороты, шумы, цветокоррекция). Увеличивает датасет в несколько раз для лучшей устойчивости модели.
3.  **`data.py`**: Формирует структуру папок YOLO (Train/Val) и создает файл конфигурации `data.yaml`.
4.  **`main.py`**: Основной конвейер. Команда `python main.py all` запустит обучение, проверку метрик, тест на папке `raw_images` и экспорт в ONNX.
5.  **`test_on_real.py`**: Финальный проверочный скрипт для тестирования весов `best.pt` на реальном изображении (`image.png`).

## 📊 Результаты
- **Архитектура:** YOLOv8 Nano (всего 6.2 МБ)
- **Точность (Precision):** 0.998 / **Полнота (Recall):** 0.997
- **Скорость:** ~60 мс/кадр на обычном процессоре.

## 🛠 Установка
1. Клонируйте репозиторий.
2. Создайте файл `.env` и вставьте: `OPENROUTER_API_KEY=ваш_ключ`.
3. Установите библиотеки: `pip install -r requirements.txt`.