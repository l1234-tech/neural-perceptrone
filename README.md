# Нейронная сеть для классификации данных

Проект реализует нейронную сеть с возможностью обучения генетическим алгоритмом и градиентным спуском. Включает инструменты для работы с данными, визуализации и оценки качества моделей.

## Структура проекта

```
ai-project/
├── CMakeLists.txt              # Конфигурация сборки CMake
├── README.md                   # Документация
├── main.cpp                    # Основная программа для демонстрации
├── requirements.txt            # Зависимости Python (для скриптов)
│
├── include/                    # Заголовочные файлы
│   ├── neural_network/
│   │   ├── NeuralNetwork.h    # Класс нейронной сети
│   │   ├── Layer.h            # Класс слоя
│   │   └── Activation.h       # Функции активации
│   │
│   ├── trainer/
│   │   ├── GeneticTrainer.h   # Генетический алгоритм обучения
│   │   ├── GradientTrainer.h  # Градиентный спуск
│   │   ├── ITrainer.h         # Интерфейс тренера
│   │   └── TrainerConfig.h    # Конфигурация обучения
│   │
│   ├── data/
│   │   ├── DataSet.h          # Структура данных
│   │   ├── DataPoint.h        # Одна точка данных
│   │   ├── DataLoader.h       # Загрузка из файла
│   │   └── XORDataGenerator.h # Генератор XOR данных
│   │
│   ├── utils/
│   │   ├── Matrix.h           # Матричные операции
│   │   ├── Vector.h           # Векторные операции
│   │   └── Random.h           # Генерация случайных чисел
│   │
│   ├── metrics/
│   │   └── ClassificationMetrics.h # Метрики классификации
│   │
│   └── visualization/
│       └── Visualizer.h        # Визуализация данных и результатов
│
├── src/                        # Реализация (cpp файлы)
│   ├── neural_network/
│   │   ├── NeuralNetwork.cpp
│   │   ├── Layer.cpp
│   │   ├── Neuron.cpp
│   │   └── Activation.cpp
│   │
│   ├── trainer/
│   │   ├── GeneticTrainer.cpp
│   │   ├── GradientTrainer.cpp
│   │   └── TrainerConfig.cpp
│   │
│   ├── data/
│   │   ├── DataSet.cpp
│   │   ├── DataPoint.cpp
│   │   ├── DataLoader.cpp
│   │   └── XORDataGenerator.cpp
│   │
│   ├── utils/
│       ├── Matrix.cpp
│       ├── Vector.cpp
│       └── Random.cpp
│   │
│   ├── metrics/
│   │   └── ClassificationMetrics.cpp
│   │
│   └── visualization/
│       └── Visualizer.cpp
│
├── tests/                      # Модульные тесты
│   ├── test_main.cpp          # Главный запускатель тестов
│   ├── test_matrix.cpp
│   ├── test_vector.cpp
│   ├── test_activation.cpp
│   ├── test_genetic_trainer.cpp
│   ├── test_gradient_trainer.cpp
│   ├── test_data_loader.cpp
│   ├── test_data_set.cpp
│   ├── test_classification_metrics.cpp
│
├── scripts/                    # Скрипты Python
│   └── run_model.py           # Скрипт для обучения и визуализации
├── images/                    # Изображения для README
│
├── data/                       # Данные для обучения
│   ├── train.csv              # Обучающая выборка
│   ├── d1.csv              
│   ├── d2.csv             
│   └── test.csv               # Тестовая выборка
│
└── .sourcecraft/               # Конфигурация SourceCraft CI/CD
    └── ci.yaml                # CI/CD для запуска тестов
```

## Сборка проекта

### Требования
- Компилятор C++ с поддержкой C++17 (g++ >= 7, clang >= 5, MSVC >= 2017)
- CMake >= 3.10
- Python 3.8+ (для скриптов визуализации)
- Библиотеки Python: numpy, matplotlib, scikit-learn (установить через `pip install -r requirements.txt`)
- Pybind11 (для Python bindings) - опционально

### Сборка с помощью CMake

```bash
# Клонирование репозитория (если нужно)
git clone <repository-url>
cd ai-project

# Создание директории сборки
mkdir build
cd build

# Конфигурация проекта
cmake ..

# Сборка
cmake --build .
```

### Сборка Python модуля (pybind11)

Для использования нейронной сети напрямую из Python необходим pybind11. Установите его через pip или систему:

```bash
# Установка pybind11 через pip (для разработки)
pip install pybind11

# Или через систему (Ubuntu/Debian)
sudo apt-get install pybind11-dev
```

Затем соберите проект с поддержкой Python модуля:

```bash
cd build
cmake .. -DPYBIND11_PYTHON_VERSION=3.8  # Укажите вашу версию Python
cmake --build .
```

После сборки в корне проекта появится файл `nnlib.so` (Linux/macOS) или `nnlib.pyd` (Windows), который можно импортировать в Python:

```python
import nnlib
```

## Использование

### Запуск основной программы

Собранная программа `main` (или `main.exe` в Windows) демонстрирует возможности библиотеки:

```bash
./build/Debug/main

или 

./build/Debug/main.exe
```

Программа выполняет:
1. Создание тестового датасета XOR
2. Обучение нейронной сети генетическим алгоритмом
3. Обучение градиентным спуском (на выбор)
4. Вывод метрик классификации (на выбор)
5. Генерацию скриптов для визуализации

### Использование Python модуля

Пример использования Python bindings:

```python
import nnlib
import random

# Создание нейронной сети
network = nnlib.create_network([2, 3, 1])  # 2 входа, 3 нейрона в скрытом слое, 1 выход

# Загрузка данных
dataset = nnlib.load_dataset("data/train.csv")

# Создание тренера
trainer = nnlib.GradientTrainer(learning_rate=0.1, epochs=100)

# Обучение
history = trainer.train(network, dataset)

# Предсказание
prediction = network.predict([0.5, -0.2])
print(f"Prediction: {prediction}")

# Дообучение на новых данных (fine-tuning)
new_data = nnlib.load_dataset("data/d1.csv")
history_continued = trainer.continue_training(network, new_data)

# Оценка метрик
metrics = nnlib.calculate_metrics(network, dataset)
print(f"Accuracy: {metrics.accuracy}, Weighted F1: {metrics.weighted_f1}")
```

### Запуск Python скрипта

Скрипт `scripts/run_model.py` предоставляет более гибкий интерфейс:

```bash
cd scripts
python run_model.py --help
```

Основные параметры:
- `--train` – путь к обучающим данным (CSV)
- `--test` – путь к тестовым данным (CSV)
- `--visualize` – включить визуализацию
- `--model` – тип модели (`genetic` или `gradient`)
- `--epochs` – количество эпох обучения

Пример:
```bash
python run_model.py --train ../data/train.csv --test ../data/test.csv --visualize --model genetic
```

## Визуализация датасетов и результатов

Проект включает модуль визуализации, который позволяет генерировать графики для анализа данных и результатов обучения.

### Генерация визуализаций

1. **2D визуализация датасета и границы решения**:
   ```cpp
   #include "visualization/Visualizer.h"
   // ...
   Visualizer::plot2D(dataset, neuralNetwork, "output_2d.png");
   ```
   Метод создаёт CSV файлы с данными и границей решения, а также Python скрипт `visualize_2d.py` для построения графика.

2. **График истории обучения**:
   ```cpp
   std::vector<double> lossHistory = trainer.train(...);
   Visualizer::plotTrainingHistory(lossHistory, "training_history.png");
   ```
   Генерирует CSV файл с историей ошибок и скрипт `visualize_history.py`.

3. **Многомерная визуализация (PCA/t-SNE)**:
   ```cpp
   Visualizer::plotMultiDimensional(dataset, neuralNetwork, "pca_visualization.png");
   ```

---

### Примеры визуализаций

Ниже приведены примеры графиков, которые можно получить с помощью библиотеки:

1. **2D датасет с границей решения** – показывает, как нейронная сеть разделяет классы в двумерном пространстве.
2. **График истории обучения** – отображает изменение ошибки в процессе обучения.
3. **Многомерная визуализация (PCA)** – проекция многомерных данных на две главные компоненты.

### 1. **d1.csv:**

   - *Иссходный датасет*:

![исходный d1](images/d1.png)

   - *Полученный моделью*:

![полученный d1](scripts/results/d1_2d_data.png)


### 2. **d2.csv:**

   - *Иссходный датасет*:

![исходный d2](images/d2.png)

   - *Полученный моделью*:

![полученный d2](scripts/results/d2_multidimensional.png)


### 3. **train.csv:**

   - *Иссходный датасет*:

![исходный train](images/train.png)

   - *Полученный моделью*:

![полученный train](scripts/results/train_2d_data.png)

---

## Тестирование

Используйте CI/CD конфигурацию SourceCraft, которая автоматически запускает тесты при каждом push и pull request.

## CI/CD

Конфигурация CI/CD находится в `.sourcecraft/ci.yaml`. При каждом изменении кода автоматически запускается сборка и тесты. Результаты можно просмотреть в интерфейсе SourceCraft.