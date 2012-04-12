#include<stdio.h>
#define CHECK_FOR_CORRECTNESS 1
#define MIN(a,b) (( (a) < (b) )?(a):(b))

/*  Kernel Function1 - Initialize the array */
__global__ void initializeArray(int* A, int* B, int N)
{
int i = threadIdx.x;

if(i<N) 
	B[i] = A[i];
printf("Setting B[%d] = %d , from A[%d] = %d\n", i, B[i], i, A[i]);
}

/* Kernel Function2 - PrefixOperations on B */
__global__ void prefixOnB(int* B, int t, int s)
{
	int i = threadIdx.x;
	B[t + i] = MIN(B[s + 2*i - 1] , B[s + 2*i]);
}

/* kernel Function3 - PrefixOperations on C */
__global__ void prefixOnC(int* B, int* C,int t, int s)
{
	int i = threadIdx.x;
	if (1 == i) 
		 C[t + i] = B[t + i];
   	else if((i%2) == 0) 
		{
		C[t + i] = C[s + (i>>1)];
		}
	else {
		C[t + i] = MIN(C[s +((i-1)>>1)] , B[t + i]);
		}
}	

/*  Kernel Function4 - Copy the results */
__global__ void copyArray(int* S, int* C, int N)
{
int i = threadIdx.x;
        S[i] = C[i];
printf("Setting S[%d] = %d , from C[%d] = %d\n", i, S[i], i, C[i]);
}


/* Compute prefix sum of A into B 
 * @param N - size of array A
 * @param A - Initial array over which prefixSum should be calculated
 * @param B - Array into which prefix Sum has to be calculated
 */
void computePrefixSum(int * A, int* B, int N)
{

}

/* Just a somple function to get log to base 2*/
int log2(int x)
{
int k = 0;
while(x>>=1) k++;
return k;
}

/* Main function - All of the implementation is in main */
int main()
{
int N = 64;

/* Declare and Initialize host arrays A, B, S */
int* h_A, *h_S, *h_B;
size_t arrSize = N*sizeof(int);
h_A = (int*)malloc(arrSize);
h_B = (int*)malloc(2*arrSize);
h_S = (int*)malloc(arrSize);


/* Declare and Initialize device arrays A, B, C, S */
int *d_A,*d_B, *d_C, *d_S;
cudaMalloc(&d_A, arrSize);
cudaMalloc(&d_B, 2*arrSize);
cudaMalloc(&d_C, 2*arrSize);
cudaMalloc(&d_S, arrSize);

int seed = 1078989; int mod = 32768, step = 7986721;

for(int i =0;i<N;i++)
	{
	h_A[i] = seed%mod;
	seed+=step;
	}
/* Copy vectors from host memory to device memory */
cudaMemcpy(d_A, h_A, arrSize, cudaMemcpyHostToDevice);

/* First call to Kernel Function to Initialize B */
int threadsPerBlock = N;
int blocksPerGrid = 1;
initializeArray<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

#ifdef CHECK_FOR_CORRECTNESS
for(int i1=0;i1<N;i1++)
	printf("DBUG_0 %d \n", h_A[i1]);

/* Checking for correctness */
cudaMemcpy(h_B, d_B, 2*arrSize, cudaMemcpyDeviceToHost);
for(int i=0;i<N;i++)
	printf("DBUG_1  %d %d\n", h_A[i], h_B[i]);
#endif

int m = N, t = 0, h=1;
int k = log2(N);
int s = 0;

for(h =1; h<=k; h++)
	{
	s = t; t += m; m >>=1;
	/* Second call to CUDA Kernel Function - This time logN calls. Every call has m parallel instances */
	blocksPerGrid = 1;
	threadsPerBlock = m;
	prefixOnB<<<blocksPerGrid, threadsPerBlock>>>(d_B, t , s);			
	}

for(h=k;h>=0;h--)
	{
	blocksPerGrid = 1;
        threadsPerBlock = m;
	
	/* Third call to kernel function - Again logN times m of them */
	prefixOnC<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, t , s); 	

	m<<=1; s= t; t-=m;
	}

/* Copy the results from C */
threadsPerBlock = N;
blocksPerGrid = 1;
copyArray<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_C, N);


cudaMemcpy(h_S, d_S, arrSize, cudaMemcpyDeviceToHost);
for(int i=0;i<N;i++)
	printf("DBUG_2 %d %d\n", h_A[i], h_S[i]);

/*Done with calculations - Free Device memory */
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
cudaFree(d_S);

/* Free host memory */
free(h_A);
free(h_B);
free(h_S);

return 0;
}


