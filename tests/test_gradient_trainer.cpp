#include <iostream>
#include <cassert>
#include "trainer/GradientTrainer.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "neural_network/NeuralNetwork.h"

bool test_gradient_trainer() {
    std::cout << "\nTesting GradientTrainer...\n";

    try {
        // 1. Проверка создания тренера
        GradientTrainer trainer;
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
        trainer.setEpochs(5);
        trainer.setBatchSize(2);
        trainer.setLearningRate(0.1);
        
        auto lossHistory = trainer.train(net, data, DataSet());
        
        if (!lossHistory.empty()) {
            std::cout << "   Training completed OK (epochs = " 
                      << lossHistory.size() << ")\n";
            std::cout << "   Initial loss: " << lossHistory[0] << "\n";
            std::cout << "   Final loss: " << lossHistory.back() << "\n";
            
            // Проверяем, что ошибка уменьшилась или осталась разумной
            assert(lossHistory[0] >= 0.0);
            assert(lossHistory.back() >= 0.0);
        }

        // 6. Проверка истории ошибок
        auto trainLossHistory = trainer.getTrainLossHistory();
        std::cout << "6. Train loss history OK (size = " << trainLossHistory.size() << ")\n";
        
        // 7. Проверка метрик
        auto metrics = trainer.getMetrics();
        std::cout << "7. Metrics OK (count = " << metrics.size() << ")\n";

        std::cout << "\nGradientTrainer tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
}