#include "utils/Matrix.h"
#include <iostream>
#include <cmath>

bool test_matrix_constructor() {
    Matrix m(3, 4);
    return m.get_stroks() == 3 && m.get_stolbs() == 4;
}

bool test_matrix_element_access() {
    Matrix m(2, 2);
    m.at(0, 0) = 1.5;
    m.at(0, 1) = 2.5;
    m.at(1, 0) = 3.5;
    m.at(1, 1) = 4.5;
    
    return m.at(0, 0) == 1.5 &&
           m.at(0, 1) == 2.5 &&
           m.at(1, 0) == 3.5 &&
           m.at(1, 1) == 4.5;
}

bool test_matrix_multiplication() {
    Matrix a(2, 3);
    a.at(0, 0) = 1; a.at(0, 1) = 2; a.at(0, 2) = 3;
    a.at(1, 0) = 4; a.at(1, 1) = 5; a.at(1, 2) = 6;
    
    Matrix b(3, 2);
    b.at(0, 0) = 7;  b.at(0, 1) = 8;
    b.at(1, 0) = 9;  b.at(1, 1) = 10;
    b.at(2, 0) = 11; b.at(2, 1) = 12;
    
    Matrix result = Matrix::multipy(a, b);
    
    return result.get_stroks() == 2 &&
           result.get_stolbs() == 2 &&
           std::abs(result.at(0, 0) - 58.0) < 1e-10 &&
           std::abs(result.at(0, 1) - 64.0) < 1e-10 &&
           std::abs(result.at(1, 0) - 139.0) < 1e-10 &&
           std::abs(result.at(1, 1) - 154.0) < 1e-10;
}

bool test_matrix() {
    std::cout << "\n=== Matrix Tests ===\n";
    
    bool all_passed = true;
    
    #define RUN_MATRIX_TEST(name) \
        std::cout << "  " #name ": "; \
        if (name()) { \
            std::cout << "OK\n"; \
        } else { \
            std::cout << "FAILED\n"; \
            all_passed = false; \
        }
    
    RUN_MATRIX_TEST(test_matrix_constructor);
    RUN_MATRIX_TEST(test_matrix_element_access);
    RUN_MATRIX_TEST(test_matrix_multiplication);
    
    return all_passed;
}