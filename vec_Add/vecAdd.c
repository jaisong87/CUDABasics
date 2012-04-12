

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
int i = threadIdx.x;
C[i] = A[i] + B[i];
}

int main()
{
int N = 256;
float* A = malloc(N*sizeof(float));
float* B = malloc(N*sizeof(float));
float* C = malloc(N*sizeof(float));
for(int i=0;i<N;i++)
	A[i] = B[i] = i;

// Kernel invocation with N threads
VecAdd<<<1, N>>>(A, B, C);

for(int i=0;i<N;i++)
	printf("%f %f %f\n", A[i], B[i], C[i]);
return 0;
}


