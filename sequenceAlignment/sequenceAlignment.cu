#include<stdio.h>
#define CHECK_FOR_CORRECTNESS 1
#define MIN(a,b) (( (a) < (b) )?(a):(b))
#define GE 1
#define GI 2

/* Following section contains Kernel functions used by prefix sum */
/*  Kernel Function1 - Initialize the array */
__global__ void initializeArray(int* A, int* B, int N)
{
int i = threadIdx.x;

if(i<N) 
	B[i] = A[i];
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
//printf("Setting S[%d] = %d , from C[%d] = %d\n", i, S[i], i, C[i]);
}

/* Just a somple function to get log to base 2*/
int log2(int x)
{
int k = 0;
while(x>>=1) k++;
return k;
}

/* Compute prefix sum of A into B 
 * @param N - size of array A
 * @param d_A - Initial device(CUDA)-array over which prefixSum should be calculated
 * @param d_S - device(CUDA)-array into which prefix Sum has to be calculated
 */
void computePrefixSum(int * d_A, int* d_S, int N)
{
int * d_B, *d_C;
size_t arrSize = N*sizeof(int);
cudaMalloc(&d_B, 2*arrSize);
cudaMalloc(&d_C, 2*arrSize);


/* First call to Kernel Function to Initialize B */
int threadsPerBlock = N;
int blocksPerGrid = 1;
initializeArray<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

/* A few variables required in prefix-computations */
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

/* Freeing two temporary device arrays B, C */
cudaFree(d_B);
cudaFree(d_C);

return;
}

/* Set of Kernel Functions used in sequence alignment calculation */

/* Kernel function to initialize d_G0, d_D0, d_I0 */
__global__ void initFirstRow(int *d_D0,int * d_I0,  int *d_G0)
{
int i = threadIdx.x;
d_G0[i] = GI + GE*i;
d_D0[i] = GE*(i+1) + GI;
if(0 == i) d_I0[i] = d_G0[i] + GE;
}

/* Kernel Function to update D from previous row */
__global__ void updateD(int* d_D1, int* d_D0, int* d_G0)
{
 int j = threadIdx.x;	
 d_D1[j] = MIN(d_D0[j] , d_G0[j] + GI )+GE;
}

/* Kernel Function to update array-U from currentD and previous G */
__global__ void updateU(int* d_U , int* d_D1, int* d_G0, int i, char* d_X, char* d_Y)
{
 int j = threadIdx.x;	
 if(j!=0)
	{
		int Sij;
		if(d_X[i] == d_Y[j]) Sij = 0;
		else Sij = 1;	
		d_U[j] = MIN(d_D1[j], d_G0[j-1] + Sij);
	}
}

/* Kernel Function to update array-V from array-U */
__global__ void updateV(int* d_V, int* d_U)
{
 int j = threadIdx.x;	
 if(j!=0)
	{	
		d_V[j] = d_U[j] + GI - j*GE;
	}
}


/* Main function - All of the implementation is in main */
int main()
{
int N;
int blocksPerGrid, threadsPerBlock;
char * X, *Y; /* char arrays in */
char * d_X, *d_Y; /* Global so that everyone can access */

/* Set of rows for matrices D, I, G and  arrays U, V  */
/* Have two versions R0, R1 for every array and they are used interchangably in every iteration */
int* d_D0, *d_D1, *d_I0, *d_I1, *d_G0, *d_G1, *d_U, *d_V, *d_S;

scanf("%d",&N);
size_t strSize = (N+1)*sizeof(char);
size_t arrSize = N*sizeof(int);

X = (char*) malloc(strSize);
Y = (char*) malloc(strSize);
printf("Going to take input for string with size %d\n", N);
scanf("%s", X);
scanf("%s", Y);

printf("%s\n", X);
printf("%s\n", Y);

/* Declare and Initialize device arrays d_X, d_Y */
cudaMalloc(&d_X, strSize );
cudaMalloc(&d_Y, strSize );

cudaMalloc(&d_D0, arrSize );
cudaMalloc(&d_D1, arrSize );
cudaMalloc(&d_G0, arrSize );
cudaMalloc(&d_G1, arrSize );
cudaMalloc(&d_I0, arrSize );
cudaMalloc(&d_I1, arrSize );
cudaMalloc(&d_U, arrSize );
cudaMalloc(&d_V, arrSize );
cudaMalloc(&d_S, arrSize );

/* Copy vectors from host memory to device memory */
cudaMemcpy(d_X, X, strSize , cudaMemcpyHostToDevice);
cudaMemcpy(d_Y, Y, strSize , cudaMemcpyHostToDevice);

/*Initialize set of rows d_G0, d_I0, d_D0 */
blocksPerGrid = 1;
threadsPerBlock = N;
initFirstRow<<<blocksPerGrid, threadsPerBlock>>>(d_D0, d_I0, d_G0);

/* For rows 1 to N calculate D, G, I from previous rows */
for(int i=1;i<N;i++)
	{
	if(i%2 == 1) /* Odd rows */
			{
	updateD<<<blocksPerGrid, threadsPerBlock>>>(d_D1, d_D0, d_G0);								
	updateU<<<blocksPerGrid, threadsPerBlock>>>(d_U , d_D1, d_G0, i, d_X, d_Y);
	updateV<<<blocksPerGrid, threadsPerBlock>>>(d_V , d_U);
	computePrefixSum(d_V, d_S, N);	
			
			}
	else /*Even rows*/
		{
	updateD<<<blocksPerGrid, threadsPerBlock>>>(d_D0, d_D1, d_G1);								
	updateU<<<blocksPerGrid, threadsPerBlock>>>(d_U , d_D0, d_G1, i, d_X, d_Y );
	updateV<<<blocksPerGrid, threadsPerBlock>>>(d_V , d_U);
	computePrefixSum(d_V, d_S, N);	
		
		}
	}

/*Done with calculations - Free Device memory */
cudaFree(d_X);
cudaFree(d_Y);

cudaFree(d_G0);
cudaFree(d_G1);
cudaFree(d_I0);
cudaFree(d_I1);
cudaFree(d_D0);
cudaFree(d_D1);
cudaFree(d_V);
cudaFree(d_U);
cudaFree(d_S);

printf("%s\n", X);
printf("%s\n", Y);

/* Free host memory */
free(X);
free(Y);

return 0;
}


