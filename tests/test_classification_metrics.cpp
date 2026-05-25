#include <iostream>
#include <cassert>
#include <cmath>
#include "metrics/ClassificationMetrics.h"
#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "data/DataPoint.h"

// Простая заглушка нейронной сети для тестирования метрик
class TestNeuralNetwork : public NeuralNetwork {
public:
    TestNeuralNetwork() : NeuralNetwork({2, 1}) {}
    
    // Всегда предсказывает 0
    int predict(const Vector& input) const {
        return 0;
    }
    
    // Заглушки для остальных методов
    Vector forward(const Vector& input) const {
        return Vector(1);
    }
    
    double computeLoss(const DataSet& data) const {
        return 0.0;
    }
    
    double computeLoss(const DataSet& data, double lambda) const {
        return 0.0;
    }
    
    std::vector<double> getWeights() const {
        return {};
    }
    
    void setWeights(const std::vector<double>& weights) {}
    
    std::vector<size_t> getArchitecture() const {
        return {2, 1};
    }
    
    size_t getNumLayers() const {
        return 2;
    }
    
    size_t getTotalParameters() const {
        return 0;
    }
};

bool test_classification_metrics() {
    std::cout << "\nTesting ClassificationMetrics...\n";

    try {
        // 1. Создание тестового датасета
        DataSet dataset;
        // Добавляем точки с известными метками
        // 3 точки класса 0, 2 точки класса 1
        dataset.add(DataPoint({0.0, 0.0}, 0));
        dataset.add(DataPoint({0.1, 0.1}, 0));
        dataset.add(DataPoint({0.2, 0.2}, 0));
        dataset.add(DataPoint({1.0, 1.0}, 1));
        dataset.add(DataPoint({1.1, 1.1}, 1));
        
        std::cout << "1. Test dataset created OK (size = " << dataset.size() << ")\n";

        // 2. Создание тестовой нейронной сети (всегда предсказывает 0)
        TestNeuralNetwork net;
        std::cout << "2. Test neural network created OK\n";

        // 3. Расчет метрик
        ClassificationMetrics metrics = calculateMetrics(net, dataset);
        std::cout << "3. calculateMetrics OK\n";
        
        // 4. Проверка базовых полей
        std::cout << "4. Checking basic fields:\n";
        std::cout << "   Accuracy: " << metrics.accuracy << "\n";
        std::cout << "   Macro F1: " << metrics.macroF1 << "\n";
        std::cout << "   Weighted F1: " << metrics.weightedF1 << "\n";
        std::cout << "   Total samples: " << metrics.totalSamples << "\n";
        
        assert(metrics.totalSamples == 5);
        assert(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert(metrics.macroF1 >= 0.0 && metrics.macroF1 <= 1.0);
        assert(metrics.weightedF1 >= 0.0 && metrics.weightedF1 <= 1.0);

        // 5. Проверка perClass метрик
        std::cout << "5. Checking per-class metrics:\n";
        assert(metrics.perClass.size() == 2); // Должны быть метрики для классов 0 и 1
        
        for (const auto& [class_label, class_metrics] : metrics.perClass) {
            std::cout << "   Class " << class_label << ":\n";
            std::cout << "     Precision: " << class_metrics.precision << "\n";
            std::cout << "     Recall: " << class_metrics.recall << "\n";
            std::cout << "     F1: " << class_metrics.f1 << "\n";
            std::cout << "     Support: " << class_metrics.support << "\n";
            
            assert(class_metrics.precision >= 0.0 && class_metrics.precision <= 1.0);
            assert(class_metrics.recall >= 0.0 && class_metrics.recall <= 1.0);
            assert(class_metrics.f1 >= 0.0 && class_metrics.f1 <= 1.0);
            assert(class_metrics.support >= 0);
        }

        // 6. Проверка printMetrics (должен работать без ошибок)
        std::cout << "6. Testing printMetrics (output should follow):\n";
        printMetrics(metrics);
        std::cout << "   printMetrics OK\n";

        // 7. Проверка calculateFinalScore
        double f1_d1 = 0.8;
        double f1_d2 = 0.6;
        double final_score = calculateFinalScore(f1_d1, f1_d2);
        std::cout << "7. calculateFinalScore OK\n";
        std::cout << "   F1(d1) = " << f1_d1 << ", F1(d2) = " << f1_d2 
                  << ", final score = " << final_score << "\n";
        
        // Проверяем формулу: 0.5 * F1(d1) + 0.5 * F1(d2)
        double expected_score = 0.5 * f1_d1 + 0.5 * f1_d2;
        assert(std::abs(final_score - expected_score) < 1e-10);

        // 8. Проверка edge cases
        // Все точки одного класса
        DataSet single_class_dataset;
        single_class_dataset.add(DataPoint({0.0, 0.0}, 0));
        single_class_dataset.add(DataPoint({0.1, 0.1}, 0));
        single_class_dataset.add(DataPoint({0.2, 0.2}, 0));
        
        ClassificationMetrics single_class_metrics = calculateMetrics(net, single_class_dataset);
        std::cout << "8. Single-class dataset metrics OK\n";
        assert(single_class_metrics.totalSamples == 3);
        
        // Пустой датасет
        DataSet empty_dataset;
        try {
            ClassificationMetrics empty_metrics = calculateMetrics(net, empty_dataset);
            std::cout << "   Empty dataset handled OK\n";
        } catch (const std::exception& e) {
            std::cout << "   Empty dataset threw exception (expected): " << e.what() << "\n";
        }

        // 9. Проверка макросов precision, recall, F1
        std::cout << "9. Checking macro averages:\n";
        std::cout << "   Macro precision: " << metrics.macroPrecision << "\n";
        std::cout << "   Macro recall: " << metrics.macroRecall << "\n";
        std::cout << "   Macro F1: " << metrics.macroF1 << "\n";
        
        assert(metrics.macroPrecision >= 0.0 && metrics.macroPrecision <= 1.0);
        assert(metrics.macroRecall >= 0.0 && metrics.macroRecall <= 1.0);
        
        // 10. Проверка weighted averages
        std::cout << "10. Checking weighted averages:\n";
        std::cout << "    Weighted precision: " << metrics.weightedPrecision << "\n";
        std::cout << "    Weighted recall: " << metrics.weightedRecall << "\n";
        std::cout << "    Weighted F1: " << metrics.weightedF1 << "\n";
        
        assert(metrics.weightedPrecision >= 0.0 && metrics.weightedPrecision <= 1.0);
        assert(metrics.weightedRecall >= 0.0 && metrics.weightedRecall <= 1.0);

        std::cout << "\nClassificationMetrics tests passed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
}