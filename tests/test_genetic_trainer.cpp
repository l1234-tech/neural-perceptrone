#include <iostream>
#include <cassert>

#include "trainer/GeneticTrainer.h"
#include "trainer/TrainerConfig.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"

void testGeneticTrainer() {

    std::cout << "Testing GeneticTrainer...\n";

    // 1. Проверка создания тренера
    GeneticTrainer trainer;
    std::cout << "1. Trainer created OK\n";

    // 2. Проверка конфигурации
    TrainerConfig config = trainer.getConfig();
    assert(config.populationSize > 0);

    std::cout << "2. Config OK (populationSize = "
              << config.populationSize << ")\n";

    // 3. Создание датасета
    DataSet data;

    data.add(DataPoint({0.1, 0.2}, 0));
    data.add(DataPoint({0.8, 0.9}, 1));
    data.add(DataPoint({0.2, 0.1}, 0));
    data.add(DataPoint({0.9, 0.8}, 1));

    std::cout << "3. Dataset created OK\n";

    // 4. Проверка evaluate
    NeuralNetwork net({2,3,1});

    double accuracy = trainer.evaluate(net, data);

    assert(accuracy >= 0.0);
    assert(accuracy <= 1.0);

    std::cout << "4. Evaluate OK (accuracy = "
              << accuracy << ")\n";

    std::cout << "GeneticTrainer tests passed!\n\n";
}