

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

#define DataType double



// Compute C = A * B where "A" is a m * n matrix and "B" is a n*k matrix. 
// numARows = m
// numAColumns = n
// numBRows = n
// numBColumns = k

// Function for timer 
double cpuSecond() 
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


__global__ void M_mul(DataType *A, DataType *B, DataType *C, int numARows,int numAColumns, int numBRows, int numBColumns)
{
  //@@ Insert code to implement matrix multiplication here

  // A columns and B rows must be equal, otherwise the multiplication can't be performed

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  if (row < numARows && col < numBColumns) // accounting for the expected matrix dimension
  {
    int sum = 0;

    for (int i = 0; i < numAColumns; i++)
    {
      sum += A[row*numAColumns + i] * B[i*numBColumns + col]; // rows are added
    }
    C[row * numBColumns + col] = sum;
  }

}

int main(int argc, char* argv[]) {

  printf ("Excecuting %s \n", argv[0] );

  int numARows = strtol(argv[1],NULL,10);    // number of rows in the matrix A
  int numAColumns = strtol(argv[2],NULL,10); // number of columns in the matrix A
  int numBRows = numAColumns;    // number of rows in the matrix B
  int numBColumns = strtol(argv[3],NULL,10); // number of columns in the matrix B
  int numCRows = numARows;
  int numCColumns = numBColumns;
  

  //@@ Insert code below to allocate Host memory for input and output

  DataType *h_dA, *h_dB, *h_dC;

  //DataType *hostA = (DataType*)malloc(sizeof(DataType)*numARows*numAColumns); // The A matrix
  //DataType *hostB = (DataType*)malloc(sizeof(DataType)*numBRows*numBColumns); // The B matrix
  //DataType *hostC = (DataType*)malloc(sizeof(DataType)*numARows*numBColumns); // The output C matrix

  cudaMallocManaged(&h_dA,sizeof(DataType)*numARows*numAColumns);
  cudaMallocManaged(&h_dB,sizeof(DataType)*numBRows*numBColumns);
  cudaMallocManaged(&h_dC,sizeof(DataType)*numARows*numBColumns);

  DataType *resultRef = (DataType*)malloc(sizeof(DataType)*numARows*numBColumns); // The reference result
  


  //@@ Insert code below to allocate GPU memory here

  //DataType *deviceA,*deviceB,*deviceC;
  //cudaMalloc(&deviceA, sizeof(DataType)*numARows*numAColumns);
  //cudaMalloc(&deviceB, sizeof(DataType)*numBRows*numBColumns);
  //cudaMalloc(&deviceC, sizeof(DataType)*numARows*numBColumns);



  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  srand((unsigned) time(NULL));

  // Matrix A initiallization
  for (int i = 0; i<numARows; ++i)
  {
    for (int j = 0; j < numAColumns; ++j)
    {
      h_dA [i* numAColumns + j] = rand() % 10; 
    }
  }

  // Matrix B
  for (int i = 0; i < numBRows; ++i)
  {
    for (int j = 0; j < numBColumns; ++j)
    {
      h_dB [i* numBColumns + j] = rand() % 10;
    }
  }

  // CPU reference result
  for (int i = 0; i < numARows; ++i)
  {
    for (int j = 0; j < numBColumns; ++j)
    {
      int tmp = 0.0;
      for (int h = 0; h < numBRows; h++)
      {
        tmp += h_dA[i*numAColumns+h] * h_dB[h*numBColumns+j];
      }
      resultRef[i*numBColumns+j] = tmp;
    }
  }


  
  //@@ Insert code to below to Copy memory to the GPU here


  //double s_HostToDevice = cpuSecond();
  //cudaMemcpy (deviceA, hostA, sizeof(DataType)*numARows*numAColumns,cudaMemcpyHostToDevice);
  //cudaMemcpy (deviceB, hostB, sizeof(DataType)*numBRows*numBColumns,cudaMemcpyHostToDevice);
  //double HostToDevice = cpuSecond() - s_HostToDevice;

  
  //@@ Initialize the grid and block dimensions here

  unsigned int g_rows = (numARows + BLOCK_SIZE - 1)/ BLOCK_SIZE; 
  unsigned int g_cols = (numBColumns + BLOCK_SIZE - 1)/ BLOCK_SIZE;

  dim3 dimGrid (g_cols,g_rows);
  dim3 dimBlock (BLOCK_SIZE,BLOCK_SIZE);


  //@@ Launch the GPU Kernel here

  

  double s_kernel = cpuSecond();
  M_mul <<<dimGrid,dimBlock>>> (h_dA,h_dB,h_dC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  double Kernel = cpuSecond() - s_kernel;


  //@@ Copy the GPU memory back to the CPU here

  //double s_DeviceToHost = cpuSecond();
  //cudaMemcpy(hostC,deviceC, sizeof(DataType)*numARows*numBColumns,cudaMemcpyDeviceToHost);
  //cudaThreadSynchronize();
  //double DeviceToHost  = cpuSecond() - s_DeviceToHost;

  //@@ Insert code below to compare the output with the reference
  
  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      if (fabs(resultRef[i*numBColumns + j] - h_dC [i*numBColumns + j]) > 1e-5)
      {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  printf("Launching with %d threads per blocks and %d x %d blocks\n",BLOCK_SIZE,g_rows,g_cols);
  printf("Time spent on kernel: %f s \n",Kernel);
  //printf("Host to device copy: %f s \n",HostToDevice);
  //printf("Device to Host copy: %f s \n",DeviceToHost);
  printf("PASSED\n");

  //@@ Free the GPU memory here

  //cudaFree(deviceA);
  //cudaFree(deviceB);
  //cudaFree(deviceC);

  //@@ Free the CPU memory here

  //free(hostA);
  //free(hostB);
  //free(hostC);
  free(resultRef);
  

  return 0;
}
