# 🚇 MetroSynth: Synthetic Dataset Generator & Damage Detector

[Russian Description](#russian-description)

**MetroSynth** is a professional framework for creating high-quality synthetic datasets and training AI models to detect interior damage in public transport. It uses generative AI to solve the "data scarcity" problem in industrial environments.

## 🌟 Key Concept
Instead of waiting for real damage to occur, this project **generates it**. By combining Large Vision Models with the YOLOv8 architecture, we can simulate thousands of repair scenarios and train a robust detector before a single real photo is even taken.

## 🛠 Features
- **Prompt-Driven Generation:** All car styles and damage types are defined in `prompts.json`.
- **AI Engine:** Defaulted to `gpt-5-image-mini` via OpenRouter, but compatible with **Flux.1**, **SDXL**, or **DALL-E**.
- **Heavy Augmentation:** Custom filters to simulate CCTV noise, poor lighting, and JPEG artifacts.
- **YOLOv8 Integration:** Fully automated training pipeline with ONNX export.

## 🚀 Execution Order
1.  **`generate_data.py`**: Generation Engine. Reads `prompts.json` and creates the `generated_dataset`.
2.  **`augment.py`**: Data Multiplier. Applies physical and digital distortions.
3.  **`data.py`**: Dataset Orchestrator. Formats data for YOLOv8 (Train/Val split).
4.  **`main.py`**: Training Pipeline. Runs training, validation, and exports to **ONNX**.
5.  **`test_on_real.py`**: Production Test. Runs the model on real photos (`image.png`).

---

<a name="russian-description"></a>

# 🚇 MetroSynth: Генератор синтетических данных и Детектор повреждений

**MetroSynth** — это фреймворк для создания высококачественных синтетических датасетов и обучения нейросетей для поиска повреждений интерьера (сиденья, пол, поручни). 

## 🌟 Концепция
Мы не ждем реальных поломок — мы **создаем их**. Используя мощь генеративного ИИ и архитектуру YOLOv8, проект позволяет имитировать тысячи сценариев износа и обучить модель еще до того, как будет собрана база реальных фотографий.

## 🏗 Особенности
- **Гибкие промты:** Описание вагонов и типов повреждений вынесено в `prompts.json`.
- **ИИ-движок:** Поддержка любой модели через OpenRouter (от `gpt-5-image-mini` до `Flux.1`).
- **Умная аугментация:** Имитация шумов камер видеонаблюдения, плохого освещения и артефактов сжатия.
- **Промышленный стандарт:** Автоматический экспорт обученной модели в **ONNX**.

## 🚀 Порядок запуска
1.  **`generate_data.py`**: Движок генерации. Читает `prompts.json` и создает базовые фото.
2.  **`augment.py`**: Множитель данных. Применяет программные фильтры и искажения.
3.  **`data.py`**: Подготовка датасета. Разделяет данные на обучение и валидацию.
4.  **`main.py`**: Конвейер обучения. Обучает YOLOv8 и делает экспорт в **ONNX**.
5.  **`test_on_real.py`**: Тест на реальных данных. Проверяет готовую модель на файле `image.png`.

## 📊 Результаты (YOLOv8 Nano)
- **Вес модели:** 6.2 МБ
- **Точность (mAP50-95):** 0.995
- **Скорость:** ~60мс (CPU)