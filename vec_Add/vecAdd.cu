#include<stdio.h>

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
int i = blockDim.x*blockIdx.x + threadIdx.x;

if(i<N) 
	C[i] = A[i] + B[i];

printf("In thread-i, we are using value %f + %f  = %f\n", A[i], B[i], C[i]);
}

int main()
{
int N = 1024;
float* h_A, *h_B, *h_C;

size_t arrSize = N*sizeof(float);

h_A = (float*)malloc(arrSize);
h_B = (float*)malloc(arrSize);
h_C = (float*)malloc(arrSize);


for(int i=0;i<N;i++)
	h_A[i] = h_B[i] = i;


float *d_A,*d_B, *d_C;
cudaMalloc(&d_A, arrSize);
cudaMalloc(&d_B, arrSize);
cudaMalloc(&d_C, arrSize);

// Copy vectors from host memory to device memory
cudaMemcpy(d_A, h_A, arrSize, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_A, arrSize, cudaMemcpyHostToDevice);

int threadsPerBlock = 256;
int blocksPerGrid = N/threadsPerBlock;
if(N%threadsPerBlock) blocksPerGrid++;

//(N + threadsPerBlock – 1) / threadsPerBlock;


VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

cudaMemcpy(h_C, d_C, arrSize, cudaMemcpyDeviceToHost);

for(int i=0;i<N;i++)
	printf("%f %f %f\n", h_A[i], h_B[i], h_C[i]);

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

return 0;
}


