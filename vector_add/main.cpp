#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include "vector_add.cu"

// Function to print a vector
void printVector(const float* vec, int n, const std::string& name) {
    std::cout << name << " = [";
    for (int i = 0; i < n; i++) {
        std::cout << std::fixed << std::setprecision(2) << vec[i];
        if (i < n - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

int main() {
    // Set random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // Define vector size
    const int n = 10;
    
    // Allocate host memory
    float* A = new float[n];
    float* B = new float[n];
    float* C = new float[n];
    
    // Initialize vectors with random values
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;
        B[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;
    }
    
    // Print input vectors
    std::cout << "Input vectors:" << std::endl;
    printVector(A, n, "A");
    printVector(B, n, "B");
    
    // Call the CUDA function to add vectors
    vecAdd(A, B, C, n);
    
    // Print result vector
    std::cout << "\nResult vector:" << std::endl;
    printVector(C, n, "C");
    
    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
} 