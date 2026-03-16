## здесь на с++ будет делаться нейронка, проект по с++ 2 сем - 1 лаба
---
## Структура проекта (черновой вариант):

ai-project/
├── CMakeLists.txt              # Файл requirements
├── README.md                   # Документация и описание
├── main.cpp                    # Точка входа, демонстрация
│
├── include/                    # Заголовочные файлы
│   │   ├── NeuralNetwork.h    # Класс нейронной сети
│   │   ├── Layer.h            # Класс слоя
│   │   ├── Neuron.h           # Класс нейрона (не факт что нужен)
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
│   │   └── DataLoader.cpp
│   │
│   └── utils/
│       ├── Matrix.cpp
│       ├── Vector.cpp
│       └── Random.cpp
│
├── tests/                      # Юнит-тесты
│   ├── test_neural_network.cpp
│   ├── test_genetic_trainer.cpp
│   └── test_data.cpp
│
├── data/                       # Данные для обучения
│   ├── train.csv              # Обучающая выборка
│   ├── test.csv               # Тестовая выборка
