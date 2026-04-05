#include "utils/Vector.h"
#include <iostream>
#include <cmath>

bool test_vector_constructor() {
    Vector v1(5);
    Vector v2(3, 2.5);
    Vector v3 = {1.0, 2.0, 3.0};
    
    return v1.size() == 5 &&
           v2.size() == 3 &&
           v2[0] == 2.5 &&
           v3.size() == 3 &&
           v3[0] == 1.0 &&
           v3[1] == 2.0 &&
           v3[2] == 3.0;
}

bool test_vector_operations() {
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};
    
    Vector sum = a + b;
    double dot = a.dot(b);
    
    return sum.size() == 3 &&
           std::abs(sum[0] - 5.0) < 1e-10 &&
           std::abs(sum[1] - 7.0) < 1e-10 &&
           std::abs(sum[2] - 9.0) < 1e-10 &&
           std::abs(dot - 32.0) < 1e-10;
}

bool test_vector() {
    std::cout << "\n=== Vector Tests ===\n";
    
    bool all_passed = true;
    
    #define RUN_VECTOR_TEST(name) \
        std::cout << "  " #name ": "; \
        if (name()) { \
            std::cout << "OK\n"; \
        } else { \
            std::cout << "FAILED\n"; \
            all_passed = false; \
        }
    
    RUN_VECTOR_TEST(test_vector_constructor);
    RUN_VECTOR_TEST(test_vector_operations);
    
    return all_passed;
}