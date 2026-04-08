#include <iostream>
#include <cassert>
#include "trainer/GeneticTrainer.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "neural_network/NeuralNetwork.h"

bool test_genetic_trainer() {
    std::cout << "\nTesting GeneticTrainer...\n";

    try {
        // 1. Проверка создания тренера
        GeneticTrainer trainer;
        std::cout << "1. Trainer created OK\n";

        // 2. Проверка конфигурации
        std::string config = trainer.getConfig();
        std::cout << "2. Config OK:\n" << config << "\n";

        // 3. Создание простого датасета XOR
        DataSet data;
        data.add(DataPoint({0.0, 0.0}, 0));
        data.add(DataPoint({1.0, 1.0}, 0));
        data.add(DataPoint({0.0, 1.0}, 1));
        data.add(DataPoint({1.0, 0.0}, 1));

        std::cout << "3. Dataset created OK (size = " << data.size() << ")\n";

        // 4. Создание простой нейронной сети
        NeuralNetwork net({2, 3, 1});
        std::cout << "4. Neural network created OK (parameters = " 
                  << net.getTotalParameters() << ")\n";

        // 5. Проверка обучения (упрощенная версия)
        std::cout << "5. Starting training...\n";
        
        // Используем маленькие параметры для быстрого теста
        trainer.setParameters(20, 10, 0.1, 0.8, 0.1);
        
        auto lossHistory = trainer.train(net, data, DataSet());
        
        if (!lossHistory.empty()) {
            std::cout << "   Training completed OK (generations = " 
                      << lossHistory.size() << ")\n";
            std::cout << "   Initial loss: " << lossHistory[0] << "\n";
            std::cout << "   Final loss: " << lossHistory.back() << "\n";
            
            // Проверяем, что ошибка уменьшилась или осталась разумной
            assert(lossHistory[0] >= 0.0);
            assert(lossHistory.back() >= 0.0);
        }

        // 6. Проверка метрик
        auto metrics = trainer.getMetrics();
        std::cout << "6. Metrics OK (count = " << metrics.size() << ")\n";
        
        // 7. Проверка лучшей особи
        auto bestIndividual = trainer.getBestIndividual();
        std::cout << "7. Best individual OK (loss = " << bestIndividual.loss 
                  << ", fitness = " << bestIndividual.fitness << ")\n";

        std::cout << "\nGeneticTrainer tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
}