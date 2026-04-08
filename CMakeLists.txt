cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkLab)

set(CMAKE_CXX_STANDARD 17)

# Пути к заголовочным файлам
include_directories(include include/utils include/data include/neural_network include/trainer)

# Библиотека (все .cpp файлы проекта)
add_library(nn_lib
    src/utils/Matrix.cpp
    src/utils/Vector.cpp
    src/utils/Random.cpp
    src/data/DataPoint.cpp
    src/data/DataSet.cpp
    src/data/DataLoader.cpp
    src/neural_network/Activation.cpp
    src/neural_network/Layer.cpp
    src/neural_network/NeuralNetwork.cpp
    src/trainer/ImprovedGeneticTrainer.cpp
    src/data/XORDataGenerator.cpp
)

# Исполняемый файл
add_executable(main main.cpp)

# Связать main с библиотекой
target_link_libraries(main nn_lib)

# Скопировать папку data в build
file(COPY data DESTINATION ${CMAKE_BINARY_DIR})