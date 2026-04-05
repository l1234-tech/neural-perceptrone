#include "neural_network/Activation.h"
#include <iostream>
#include <cmath>
#include <vector>

bool test_sigmoid() {
    // Основные свойства sigmoid
    double s0 = Activation::sigmoid(0.0);
    if (std::abs(s0 - 0.5) > 1e-10) {
        std::cout << "FAIL: sigmoid(0) = " << s0 << ", expected 0.5\n";
        return false;
    }
    
    // Должно быть между 0 и 1
    double s10 = Activation::sigmoid(10.0);
    if (s10 <= 0.9999) {
        std::cout << "FAIL: sigmoid(10) = " << s10 << ", expected > 0.9999\n";
        return false;
    }
    
    double s_minus10 = Activation::sigmoid(-10.0);
    if (s_minus10 >= 0.0001) {
        std::cout << "FAIL: sigmoid(-10) = " << s_minus10 << ", expected < 0.0001\n";
        return false;
    }
    
    // Проверка монотонности
    if (Activation::sigmoid(1.0) <= Activation::sigmoid(0.0)) {
        std::cout << "FAIL: sigmoid should be monotonic\n";
        return false;
    }
    
    // Проверка симметрии
    double x = 3.7;
    double diff = std::abs(Activation::sigmoid(x) + Activation::sigmoid(-x) - 1.0);
    if (diff > 1e-10) {
        std::cout << "FAIL: sigmoid symmetry broken, diff = " << diff << "\n";
        return false;
    }
    
    return true;
}

bool test_relu() {
    return Activation::relu(5.0) == 5.0 &&
           Activation::relu(0.0) == 0.0 &&
           Activation::relu(-1.0) == 0.0;
}

bool test_step() {
    return Activation::step(0.6) == 1 &&
           Activation::step(0.5) == 1 &&
           Activation::step(0.4) == 0 &&
           Activation::step(2.0, 1.5) == 1 &&
           Activation::step(1.0, 1.5) == 0;
}

bool test_activation() {
    std::cout << "\n=== Activation Tests ===\n";
    
    bool all_passed = true;
    
    #define RUN_ACTIVATION_TEST(name) \
        std::cout << "  " #name ": "; \
        if (name()) { \
            std::cout << "OK\n"; \
        } else { \
            std::cout << "FAILED\n"; \
            all_passed = false; \
        }
    
    RUN_ACTIVATION_TEST(test_sigmoid);
    RUN_ACTIVATION_TEST(test_relu);
    RUN_ACTIVATION_TEST(test_step);
    
    return all_passed;
}