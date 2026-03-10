#pragma once
#include <vector>

class Activation {
public:
    // Сигмоида: сжимает любое число в диапазон (0, 1)
    // Формула: 1 / (1 + e^(-x))
    // Используется как выходная функция
    static double sigmoid(double x);

    // relu: обнуляет отрицательные, остальное без изменений
    // Формула: max(0, x)
    // Основная функция для скрытых слоёв
    static double relu(double x);

    // Пороговая функция: жёсткое решение — 0 или 1
    // Если x >= threshold → 1, иначе → 0
    static int step(double x, double threshold = 0.5);

    // Индекс максимального элемента в векторе
    static int argmax(const std::vector<double>& values);
};
