#include <cuda_runtime.h>
#include <stdio.h>

// normalized QK^T kernel
__global__ void normalized_QKT(float *Q_d, float *K_d, float *QKT, int M, int N, int d) {
    // computing row and column indices
    int row =  blockIdx.y * blockDim.y + threadIdx.y;
    int col =  blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < N)) {
        float thread_sum = 0.0;
        for (int k = 0; k < d; k++){
            thread_sum += Q_d[row * d + k] * K_d[col * d + k];
        }
        QKT[row * N + col] = thread_sum / sqrt((float)d);
    }
}

// softmax kernel
__global__ void softmax_kernel(float *QKT, float *softmax_QKT, int M, int N) {
    // one thread processes one row in the matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M) {
        // compute max value for numerical stability
        float max_val = -INFINITY;
        for (int col = 0; col < N; col++) {
            max_val = max(max_val, QKT[row * N + col]);
        }

        // compute exp(x - max_value) and sum
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            softmax_QKT[row * N + col] = exp(QKT[row * N + col] - max_val);
            sum += softmax_QKT[row * N + col];
        }

        // Normalize row by it's sum
        for (int col = 0; col < N; col++) {
            softmax_QKT[row * N + col] /= sum;
        }
    }
}

// final attention score kernel
__global__ void final_attention_computation(float *softmax_QKT, float *V_d, float *output_d, int M, int N, int d) {
    // computing row and column indices
    int row =  blockIdx.y * blockDim.y + threadIdx.y;
    int col =  blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < d)) {
        float sum = 0.0;
        for (int k = 0; k < N; k++){
            sum += softmax_QKT[row * N + k] * V_d[k * d + col];
        }
        output_d[row * d + col] = sum;
    }
}

// Helper function to print a matrix
void printMatrix(const char* name, const float* matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Q (Mxd), K (Nxd), V (Nxd), output are device pointers
void solve(const float *Q, const float *K, const float *V, float *output, int M, int N, int d) {
    // declare device pointers
    float *Q_d, *K_d, *V_d;
    float *QKT, *softmax_QKT;
    float *output_d;
    
    // For debugging
    float *QKT_host, *softmax_QKT_host;
    QKT_host = (float*)malloc(M * N * sizeof(float));
    softmax_QKT_host = (float*)malloc(M * N * sizeof(float));

    // allocate memory for input matrices -- Q, K and V
    cudaMalloc((void **) &Q_d, M * d * sizeof(float));
    cudaMalloc((void **) &K_d, N * d * sizeof(float));
    cudaMalloc((void **) &V_d, N * d * sizeof(float));

    // allocate memory for intermediate computation matrices -- QKT and softmax
    cudaMalloc((void **) &QKT, M * N * sizeof(float));
    cudaMalloc((void **) &softmax_QKT, M * N * sizeof(float));

    // allocate memory for output matrix -- output
    cudaMalloc((void **) &output_d, M * d * sizeof(float));

    // copy Q, K and V from host to device
    cudaMemcpy(Q_d, Q, M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Print input matrices for debugging
    printf("Input dimensions: M=%d, N=%d, d=%d\n", M, N, d);
    // printMatrix("Q", Q, M, d);
    // printMatrix("K", K, N, d);
    // printMatrix("V", V, N, d);

    // LAUNCH KERNELS
    // define grid and block dimensions for normalized_QKT kernel and call it
    dim3 dimGridNormalizedQKT((int)ceilf(N/32.0f), (int)ceilf(M/32.0f), 1);
    dim3 dimBlockNormalizedQKT(32, 32, 1);
    normalized_QKT<<<dimGridNormalizedQKT, dimBlockNormalizedQKT>>>(Q_d, K_d, QKT, M, N, d);  
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in normalized_QKT kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy QKT to host for debugging
    cudaMemcpy(QKT_host, QKT, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix("QKT", QKT_host, M, N);

    // define grid and block dimensions for softmax kernel and call it
    dim3 dimGridSoftmax(1, (int)ceilf(M/32.0f), 1);  // One block per 32 rows; M rows in total -- MxN matrix 
    dim3 dimBlockSoftmax(1, 32, 1); // 32 threads per block 
    softmax_kernel<<<dimGridSoftmax, dimBlockSoftmax>>>(QKT, softmax_QKT, M, N);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy softmax_QKT to host for debugging
    cudaMemcpy(softmax_QKT_host, softmax_QKT, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix("softmax_QKT", softmax_QKT_host, M, N);

    // define grid and block dimensions for final attention computation kernel and call it
    dim3 dimGridFinalAttention((int)ceilf(d/32.0f), (int)ceilf(M/32.0f), 1);
    dim3 dimBlockFinalAttention(32, 32, 1);
    final_attention_computation<<<dimGridFinalAttention, dimBlockFinalAttention>>>(softmax_QKT, V_d, output_d, M, N, d);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in final_attention_computation: %s\n", cudaGetErrorString(err));
        return;
    }

    // free cuda memory
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(QKT);
    cudaFree(softmax_QKT);
    cudaFree(output_d);

    // copy output from device to host
    cudaMemcpy(output, output_d, M * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free debugging memory
    free(QKT_host);
    free(softmax_QKT_host);
}