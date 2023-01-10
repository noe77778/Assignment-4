#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>


__global__ void vectorAdd(float *A, float *B, float *C, int len)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < len) C[id] = A[id] + B[id];
}


// Function for timer 
double cpuSecond() 
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main(int argc, char* argv[])
{
    printf ("Excecuting %s \n", argv[0] );
    int N = strtol(argv[1],NULL,10);
    size_t size = N * sizeof(float);

    printf("[Vector addition of %d elements]\n", N);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size); // Host input A
    float *h_B = (float *)malloc(size); // Host input B
    float *h_C = (float *)malloc(size); // Host output C
    float *result_Ref = (float*)malloc(size); // Host result reference

    float *d_A = NULL; // pointer for device
    float *d_B = NULL; // pointer for device
    float *d_C = NULL; // pointer for device

    // Initialize the host input vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        result_Ref[i] = h_A[i] + h_B [i];
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    int Db = 256;
    int Dg =(N + Db - 1) / Db;
    printf("Kernel will launch with %d blocks and %d threads per block\n", Dg, Db);

    
    double totaltimestart = cpuSecond();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Kernel launching

    
    vectorAdd<<<Dg, Db>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    

    
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    

    double totaltime = cpuSecond() - totaltimestart; // timer

    printf("Total time elapsed: %f s\n", totaltime);

    
    
    for (int i = 0; i < N; ++i)
    {
        if (fabs(result_Ref[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("PASSED\n");


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
