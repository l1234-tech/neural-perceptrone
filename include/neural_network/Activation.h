#pragma once
#include <vector>

class Activation {
public:
    // Сигмоида: сжимает любое число в диапазон (0, 1)
    // Формула: 1 / (1 + e^(-x))
    // Используется для бинарной классификации и как выходная функция
    static double sigmoid(double x);

    // ReLU (Rectified Linear Unit): обнуляет отрицательные, остальное без изменений
    // Формула: max(0, x)
    // Основная функция для скрытых слоёв — быстрая и эффективная
    static double relu(double x);

    // Пороговая функция: жёсткое решение — 0 или 1
    // Если x >= threshold → 1, иначе → 0
    static int step(double x, double threshold = 0.5);

    // Индекс максимального элемента в векторе
    // Нужен для мультиклассовой классификации —
    // выбирает класс с наибольшей «уверенностью»
    static int argmax(const std::vector<double>& values);
};
