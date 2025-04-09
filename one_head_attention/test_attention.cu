#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "softmax_attention.cu"

int main() {
    // Test case from the forum
    const int M = 2;  // Number of rows in Q
    const int N = 3;  // Number of rows in K and V
    const int d = 4;  // Dimension of each vector
    
    // Allocate memory for input matrices
    float* Q = (float*)malloc(M * d * sizeof(float));
    float* K = (float*)malloc(N * d * sizeof(float));
    float* V = (float*)malloc(N * d * sizeof(float));
    float* output = (float*)malloc(M * d * sizeof(float));
    float* expected = (float*)malloc(M * d * sizeof(float));
    
    // Initialize Q
    Q[0] = 1.0f; Q[1] = 0.0f; Q[2] = 0.0f; Q[3] = 0.0f;
    Q[4] = 0.0f; Q[5] = 1.0f; Q[6] = 0.0f; Q[7] = 0.0f;
    
    // Initialize K
    K[0] = 1.0f; K[1] = 0.0f; K[2] = 0.0f; K[3] = 0.0f;
    K[4] = 0.0f; K[5] = 1.0f; K[6] = 0.0f; K[7] = 0.0f;
    K[8] = 0.0f; K[9] = 0.0f; K[10] = 1.0f; K[11] = 0.0f;
    
    // Initialize V
    V[0] = 1.0f; V[1] = 2.0f; V[2] = 3.0f; V[3] = 4.0f;
    V[4] = 5.0f; V[5] = 6.0f; V[6] = 7.0f; V[7] = 8.0f;
    V[8] = 9.0f; V[9] = 10.0f; V[10] = 11.0f; V[11] = 12.0f;
    
    // Initialize expected output
    expected[0] = 4.2888236f; expected[1] = 5.2888236f; expected[2] = 6.2888236f; expected[3] = 7.2888236f;
    expected[4] = 5.0f; expected[5] = 6.0f; expected[6] = 7.0f; expected[7] = 8.0f;
    
    // Print input matrices
    printf("Test case:\n");
    printMatrix("Q", Q, M, d);
    printMatrix("K", K, N, d);
    printMatrix("V", V, N, d);
    
    // Call the attention function
    solve(Q, K, V, output, M, N, d);
    
    // Print output and expected output
    printMatrix("Output", output, M, d);
    printMatrix("Expected", expected, M, d);
    
    // Calculate and print the maximum difference
    float max_diff = 0.0f;
    for (int i = 0; i < M * d; i++) {
        float diff = fabs(output[i] - expected[i]);
        max_diff = (diff > max_diff) ? diff : max_diff;
    }
    printf("Max difference: %.6f\n", max_diff);
    
    // Free memory
    free(Q);
    free(K);
    free(V);
    free(output);
    free(expected);
    
    return 0;
} 