#include <iostream>
#include <cassert>

// Объявления тестовых функций
bool test_matrix();
bool test_vector();
bool test_activation();

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
    
    std::cout << "\nResults: " << passed << "/" << total << " tests passed\n";
    return (passed == total) ? 0 : 1;
}