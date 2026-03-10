#include "utils/Random.h"
#include "utils/Matrix.h"
#include "utils/Vector.h"
#include "neural_network/Activation.h"
#include "neural_network/Layer.h"
#include "data/DataLoader.h"
#include "data/DataSet.h"
#include <iostream>
#include <iomanip>      
#include <string>
#include <vector>

int main() {
    initRandom(42);
    
    std::cout << "=== Neural Network Lab - Part 1 Demo ===\n\n";
    
    // 1. Тест утилит
    std::cout << "[1] Utils:\n";
    
    Matrix m(2, 2);
    m.at(0,0) = 1.0;
    m.at(0,1) = 2.0;
    m.at(1,0) = 3.0; 
    m.at(1,1) = 4.0;
    std::cout << "- Matrix 2x2: OK\n";
    std::cout << "- Transpose: [" << m.transpose().at(0,0) << ", " << m.transpose().at(0,1) << "]\n";
    
    Vector v1({1.0, 2.0}), v2({3.0, 4.0});
    std::cout << "- Vector dot product: " << v1.dot(v2) << "\n";
    std::cout << "- Activation sigmoid(2.0): " << Activation::sigmoid(2.0) << "\n\n";
    
    // 2. Тест слоя
    std::cout << "[2] Layer (2->4):\n";
    Layer hidden(2, 4);
    Vector input({0.3, 0.8});
    Vector output = hidden.forward(input);
    
    std::cout << "- Input:  [" << input[0] << ", " << input[1] << "]\n";
    std::cout << "- Output: [";
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << output[i];
        if (i < output.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Тест интерфейса для ГА
    auto weights = hidden.getWeights();
    weights[0] += randomDouble(-0.1, 0.1);
    hidden.setWeights(weights);
    std::cout << "- GA interface (get/set weights): OK\n\n";
    
    // 3. Тест данных
    std::cout << "[3] DataLoader:\n";
    try {
        DataSet trainData = DataLoader::loadFromCSV("data/train.csv");
        std::cout << "- Loaded " << trainData.size() << " training points\n";
        
        if (!trainData.empty()) {
            const auto& p = trainData.get(0);
            const auto& f = p.getFeatures();
            std::cout << "- Sample: [" << f[0] << ", " << f[1] << "] -> class " << p.getLabel() << "\n";
        }
        
        DataSet predictData = DataLoader::loadFromCSVNoLabels("data/test.csv");
        std::cout << "- Loaded " << predictData.size() << " points for prediction\n";
    } catch (...) {
        std::cout << "- Data files not found (optional)\n";
    }
}