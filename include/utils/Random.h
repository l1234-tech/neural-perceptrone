#pragma once

// Инициализация (вызвать один раз в main) - здесь можем менять seed
void initRandom(unsigned seed = 42);

// Вещественное число в диапазоне [min, max]
double randomDouble(double min, double max);

// Целое число в диапазоне [min, max]
int randomInt(int min, int max);