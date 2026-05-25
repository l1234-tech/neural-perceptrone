#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include "visualization/Visualizer.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"
#include "neural_network/NeuralNetwork.h"

bool test_visualization() {
    std::cout << "\nTesting Visualizer...\n";

    try {
        // 1. Создание простого датасета XOR (2D)
        DataSet data;
        data.add(DataPoint({0.0, 0.0}, 0));
        data.add(DataPoint({1.0, 1.0}, 0));
        data.add(DataPoint({0.0, 1.0}, 1));
        data.add(DataPoint({1.0, 0.0}, 1));

        std::cout << "1. Dataset created OK (size = " << data.size() << ")\n";

        // 2. Создание простой нейронной сети
        NeuralNetwork net({2, 3, 1});
        std::cout << "2. Neural network created OK (parameters = " 
                  << net.getTotalParameters() << ")\n";

        // 3. Тест plot2D
        std::string output2d = "test_visualization_2d.png";
        std::cout << "3. Testing plot2D...\n";
        Visualizer::plot2D(data, net, output2d);
        std::cout << "   plot2D completed OK\n";
        
        // Проверяем, что скрипт создан
        std::ifstream script("visualize_2d.py");
        assert(script.good());
        script.close();
        std::cout << "   Visualization script created OK\n";

        // Проверяем, что CSV файлы созданы (они имеют фиксированные имена)
        std::ifstream points("visualization_points.csv");
        assert(points.good());
        points.close();
        std::ifstream boundary("visualization_boundary.csv");
        assert(boundary.good());
        boundary.close();
        std::cout << "   CSV files created OK\n";

        // 4. Тест plotTrainingHistory
        std::vector<double> history = {0.9, 0.7, 0.5, 0.3, 0.2};
        std::string outputHistory = "test_visualization_history.png";
        std::cout << "4. Testing plotTrainingHistory...\n";
        Visualizer::plotTrainingHistory(history, outputHistory);
        std::cout << "   plotTrainingHistory completed OK\n";

        // Проверяем, что скрипт создан
        std::ifstream scriptHist("visualize_history.py");
        assert(scriptHist.good());
        scriptHist.close();
        std::cout << "   Training history script created OK\n";

        // Проверяем, что CSV файл истории создан
        std::ifstream histCsv("training_history.csv");
        assert(histCsv.good());
        histCsv.close();
        std::cout << "   Training history CSV created OK\n";

        // 5. Очистка временных файлов (опционально)
        // В CI они останутся в папке build, что не страшно

        std::cout << "\nVisualizer tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
}