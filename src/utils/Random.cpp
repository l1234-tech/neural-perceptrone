#include "Random.h"
#include <random>

// Статический генератор (виден только в этом файле)
static std::mt19937 gen(42);

void initRandom(unsigned seed) {
    gen.seed(seed);  // Меняем seed при необходимости
}

double randomDouble(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

int randomInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}