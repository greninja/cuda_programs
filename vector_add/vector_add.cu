// kernel function to add two vectors

__global__ void vecAddKernel(float* A, float* B, float* C, int n) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) { 
        C[i] = A[i] + B[i];
    } 
}

void vecAdd(float* A, float* B, float* C, int n) { 
    float *A_d, *B_d, *C_d; 
    int size = n * sizeof(float); 
    
    // allocate memory on the device
    cudaMalloc((void **) &A_d, size); 
    cudaMalloc((void **) &B_d, size); 
    cudaMalloc((void **) &C_d, size); 
    
    // copy input data from host to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice); 
    
    // launch the kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n); 

    // copy result from device to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost); 
    
    // free memory on the device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
} 