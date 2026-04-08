## Команда ГиКи: нейронная сеть , 1 + 2 лабы
---
## Структура проекта:

ai-project/
├── CMakeLists.txt              # Файл requirements
├── README.md                   # Документация и описание
├── main.cpp
│
├── include/                    # Заголовочные файлы
│   ├── neural_network/
│   │   ├── NeuralNetwork.h    # Класс нейронной сети
│   │   ├── Layer.h            # Класс слоя
│   │   └── Activation.h       # Функции активации
│   │
│   ├── trainer/
│   │   ├── GeneticTrainer.h   # Генетический алгоритм обучения
│   │   ├── ITrainer.h         # Интерфейс тренера (абстрактный класс)
│   │   └── TrainerConfig.h    # Конфигурация обучения
│   │
│   ├── data/
│   │   ├── DataSet.h          # Структура данных
│   │   ├── DataPoint.h        # Одна точка данных
│   │   └── DataLoader.h       # Загрузка из файла
│   │   └── XORDataGenerator.h       # создание точек
│   │
│   └── utils/
│       ├── Matrix.h           # Матричные операции
│       ├── Vector.h           # Векторные операции
│       └── Random.h           # Генерация случайных чисел
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
│   │   └── TrainerConfig.cpp
│   │
│   ├── data/
│   │   ├── DataSet.cpp
│   │   ├── DataPoint.cpp
│   │   └── DataLoader.cpp
│   │   └── XORDataGenerator.cpp
│   │
│   └── utils/
│       ├── Matrix.cpp
│       ├── Vector.cpp
│       └── Random.cpp
│
├── tests/
│   ├── test_genetic_trainer.cpp
│   ├── test_activation.cpp
│   ├── test_vector.cpp
│   ├── test_matrix.cpp
│   ├── test_main.cpp
│
├── data/                       # Данные для обучения
│   ├── train.csv              # Обучающая выборка
│   ├── test.csv               # Тестовая выборка
├── .sourcecraft/                       # Данные для обучения
│   ├── ci.yaml              # ci/cd для запуска тестов при пуше в майн или пул реквест
