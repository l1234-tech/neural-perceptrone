#include <iostream>
#include <cassert>

// Объявления тестовых функций
bool test_matrix();
bool test_vector();
bool test_activation();
bool test_genetic_trainer();
bool test_gradient_trainer();
bool test_data_loader();
bool test_data_set();
bool test_classification_metrics();

int main() {
    std::cout << "Running tests...\n";
    
    int passed = 0;
    int total = 0;
    
    #define RUN_TEST(func) \
        std::cout << "Testing " #func "... "; \
        total++; \
        if (func()) { \
            std::cout << "PASSED\n"; \
            passed++; \
        } else { \
            std::cout << "FAILED\n"; \
        }
    
    RUN_TEST(test_matrix);
    RUN_TEST(test_vector);
    RUN_TEST(test_activation);
    RUN_TEST(test_genetic_trainer);
    RUN_TEST(test_gradient_trainer);
    RUN_TEST(test_data_loader);
    RUN_TEST(test_data_set);
    RUN_TEST(test_classification_metrics);
    
    std::cout << "\nResults: " << passed << "/" << total << " tests passed\n";
    return (passed == total) ? 0 : 1;
}