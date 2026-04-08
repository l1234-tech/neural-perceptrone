#include "utils/Random.h"
#include "data/DataLoader.h"
#include "data/DataSet.h"
#include "neural_network/NeuralNetwork.h"
#include "trainer/GeneticTrainer.h"
#include "trainer/TrainerConfig.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

int main()
{
    initRandom(42);

    std::cout<<"\n=== Genetic training demo ===\n";

    DataSet trainData =
        DataLoader::loadFromCSV("data/train.csv");

    DataSet testData =
        DataLoader::loadFromCSV("data/test.csv");

    std::cout<<"Train size: "
             <<trainData.size()<<std::endl;

    std::cout<<"Test size: "
             <<testData.size()<<std::endl;


    NeuralNetwork net({2,4,2});

    std::cout<<"Network created\n";

    std::cout<<"Weights: "
             <<net.getTotalWeights()
             <<std::endl;


    TrainerConfig config;

    config.populationSize = 50;
    config.generations = 30;
    config.mutationRate = 0.02;

    GeneticTrainer trainer(config);

    std::cout<<"Population: "
             <<config.populationSize
             <<std::endl;

    std::cout<<"Generations: "
             <<config.generations
             <<std::endl;


    std::cout<<"Training...\n";

    trainer.train(net,trainData);


    double trainAccuracy =
        trainer.evaluate(net,trainData);

    double testAccuracy =
        trainer.evaluate(net,testData);


    std::cout<<std::fixed
             <<std::setprecision(2);

    std::cout<<"Train accuracy: "
             <<trainAccuracy*100
             <<"%\n";

    std::cout<<"Test accuracy: "
             <<testAccuracy*100
             <<"%\n";


    const auto& history =
        trainer.getTrainingHistory();

    if(!history.empty())
    {
        double best =
            *std::max_element(
                history.begin(),
                history.end()
            );

        std::cout<<"Best accuracy: "
                 <<best*100
                 <<"%\n";
    }

    return 0;
}