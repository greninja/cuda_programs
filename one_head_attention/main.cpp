#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "solve.h"

// Helper function to print a matrix
void printMatrix(const float* matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, 5); i++) {
        for (int j = 0; j < std::min(cols, 5); j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        if (cols > 5) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 5) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

int main() {
    // Define dimensions
    int M = 64;  // Sequence length for queries
    int N = 64;  // Sequence length for keys/values
    int d = 32;  // Embedding dimension
    
    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Allocate memory for input matrices
    std::vector<float> Q(M * d);
    std::vector<float> K(N * d);
    std::vector<float> V(N * d);
    std::vector<float> output(M * d, 0.0f);
    
    // Initialize matrices with random values
    for (int i = 0; i < M * d; i++) {
        Q[i] = dist(gen);
    }
    
    for (int i = 0; i < N * d; i++) {
        K[i] = dist(gen);
        V[i] = dist(gen);
    }
    
    // Print input matrices (or portions of them)
    printMatrix(Q.data(), M, d, "Q");
    printMatrix(K.data(), N, d, "K");
    printMatrix(V.data(), N, d, "V");
    
    std::cout << "Computing attention..." << std::endl;
    
    // Call the CUDA attention implementation
    solve(Q.data(), K.data(), V.data(), output.data(), M, N, d);
    
    // Print the output matrix
    printMatrix(output.data(), M, d, "Output");
    
    std::cout << "Attention computation completed successfully!" << std::endl;
    
    return 0;
}

// // main
// int main() {
//     // Example dimensions
//     int M = 64;  // Number of queries
//     int N = 64;  // Number of key-value pairs
//     int d = 32;  // Embedding dimension

//     // Allocate host memory
//     float *Q = (float*)malloc(M * d * sizeof(float));
//     float *K = (float*)malloc(N * d * sizeof(float));
//     float *V = (float*)malloc(N * d * sizeof(float));
//     float *output = (float*)malloc(M * d * sizeof(float));

//     // Initialize input matrices with some test values
//     for(int i = 0; i < M * d; i++) Q[i] = 1.0f;
//     for(int i = 0; i < N * d; i++) K[i] = 1.0f;
//     for(int i = 0; i < N * d; i++) V[i] = 1.0f;

//     // Call the solve function
//     solve(Q, K, V, output, M, N, d);

//     // Print some output values
//     printf("First few output values:\n");
//     for(int i = 0; i < 5; i++) {
//         printf("%f ", output[i]);
//     }
//     printf("\n");

//     // Free host memory
//     free(Q);
//     free(K);
//     free(V);
//     free(output);

//     return 0;
// }
