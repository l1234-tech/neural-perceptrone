#pragma once
#include <cstddef>

struct TrainerConfig {
    // Основные параметры (по умолчанию)
    size_t populationSize = 100;
    size_t generations = 50;
    double mutationRate = 0.01;
    double crossoverRate = 0.7;
    double elitismRate = 0.1;
    double mutationStrength = 0.1;
    size_t tournamentSize = 5;
    
    // Структура для предустановок
    struct Preset {
        size_t populationSize;
        size_t generations;
        double mutationRate;
        double crossoverRate;
        double elitismRate;
        double mutationStrength;
        size_t tournamentSize;
    };
    
    // Конструктор из Preset
    TrainerConfig(const Preset& preset);
    
    // Статические методы для предустановок
    static TrainerConfig fastConfig();
    static TrainerConfig preciseConfig();
    
    // Метод валидации
    bool validate() const;
};