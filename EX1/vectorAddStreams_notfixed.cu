
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__global__ void vectorAdd(float *A, float *B, float *C, int len, int offset)
{
    int id = offset +  blockDim.x * blockIdx.x + threadIdx.x;
    if (id < len + offset) C[id] = A[id] + B[id];
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
    
    
    const int nStreams = 4; 
    const int S_seg = N/4;
    
    size_t streambytes = S_seg * sizeof(float);

    printf("[Vector addition of %d elements]\n", N);

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;  

    checkCuda(cudaMallocHost ((void**) &h_A, size));
    checkCuda(cudaMallocHost ((void**) &h_B, size));
    checkCuda(cudaMallocHost ((void**) &h_C, size));    

    float *result_Ref = (float*)malloc(size); // Host result reference 
    float *h_check = (float*)malloc(size); // Check

    float *d_A, *d_B, *d_C;

    checkCuda(cudaMalloc((void**)&d_A, size));
    checkCuda(cudaMalloc((void**)&d_B, size));
    checkCuda(cudaMalloc((void**)&d_C, size));

    // Initialize the host input vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        result_Ref[i] = h_A[i] + h_B [i];
    }




    cudaStream_t stream[nStreams];

    for (int i = 0; i < nStreams; i++)
    {
      checkCuda(cudaStreamCreate(&stream[i]));
    }
    

    // Asynchronus version //

    int Db = 256;
    int Dg =(S_seg + Db - 1) / Db;

    printf("Kernel will launch with %d blocks and %d threads per block\n", Dg, Db);

    double totaltimestart = cpuSecond();

    //////////////////////

    for (int i = 0; i < nStreams; i++)
    {
      
      int offset = i * S_seg;
      
      checkCuda(cudaMemcpyAsync(&d_A[offset], &h_A[offset], streambytes, cudaMemcpyHostToDevice, stream[i]));
      checkCuda(cudaMemcpyAsync(&d_B[offset], &h_B[offset], streambytes, cudaMemcpyHostToDevice, stream[i]));
      vectorAdd<<<Dg, Db,0,stream[i]>>>(d_A, d_B, d_C, N, offset);
      checkCuda(cudaMemcpyAsync(&h_C[offset], &d_C[offset], streambytes, cudaMemcpyDeviceToHost, stream[i]));
      
    }

    ////////////////////////
    double totaltime = cpuSecond() - totaltimestart; // timer
    

    printf("Stream size %d and number of streams %d\n", S_seg,nStreams);


       
    
    for (int i = 0; i < N; ++i)
    {
        //printf ("REF: %f\n",result_Ref[i]);
        //printf ("CUDA: %f\n",h_C[i]);
        if (fabs(result_Ref[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("PASSED\n");

    printf("Total time elapsed: %f ms\n", totaltime);

       for (int i = 0; i < nStreams; i++)
    {
      checkCuda(cudaStreamDestroy(stream[i]));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
