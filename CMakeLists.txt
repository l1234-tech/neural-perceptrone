cmake_minimum_required(VERSION 3.15)

project(NeuralNetworkLab)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

# Пути к заголовочным файлам
include_directories(include include/utils include/data include/neural_network include/trainer include/metrics include/visualization)

# Библиотека (все .cpp файлы проекта)
add_library(nn_lib
    src/utils/Matrix.cpp
    src/utils/Vector.cpp
    src/utils/Random.cpp
    src/data/DataPoint.cpp
    src/data/DataSet.cpp
    src/data/DataLoader.cpp
    src/data/XORDataGenerator.cpp
    src/neural_network/Activation.cpp
    src/neural_network/Layer.cpp
    src/neural_network/NeuralNetwork.cpp
    src/trainer/GeneticTrainer.cpp
    src/trainer/GradientTrainer.cpp
    src/trainer/TrainerConfig.cpp
    src/metrics/ClassificationMetrics.cpp
    src/visualization/Visualizer.cpp
)

# Основной исполняемый файл
add_executable(main main.cpp)

# Связать main с библиотекой
target_link_libraries(main nn_lib)

# Исполняемый файл для тестов
add_executable(test_runner
    tests/test_main.cpp
    tests/test_matrix.cpp
    tests/test_vector.cpp
    tests/test_activation.cpp
    tests/test_genetic_trainer.cpp
    tests/test_gradient_trainer.cpp
    tests/test_data_loader.cpp
    tests/test_data_set.cpp
    tests/test_classification_metrics.cpp
)

# Связать тесты с библиотекой
target_link_libraries(test_runner nn_lib)

# Скопировать папку data в build
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})

# 1. Подключаем механизм загрузки внешних зависимостей
include(FetchContent)

# 2. Настраиваем загрузку pybind11 из официального репозитория
# Используем стабильную версию v2.10.4 или новее (v2.11.1 рекомендуется)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)

# 3. Скачиваем и делаем доступным (это создаст цель pybind11::module)
# Важно: это нужно делать ПОСЛЕ объявления всех твоих библиотек (nn_lib), 
# но ДО создания модуля nnlib
FetchContent_MakeAvailable(pybind11)

# 4. Создаем Python-модуль
# pybind11_add_module - это специальная функция, которая сама настроит компилятор
pybind11_add_module(nnlib src/python_bindings.cpp)

# 5. Линкуем наш модуль с основной логикой нейросети
target_link_libraries(nnlib PRIVATE nn_lib)

# 6. Копируем собранный модуль (.pyd или .so) в корень проекта для удобства импорта в Python
add_custom_command(TARGET nnlib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:nnlib>
        ${CMAKE_SOURCE_DIR}/nnlib.pyd
    COMMENT "Copying Python module (nnlib.pyd) to project root"
)