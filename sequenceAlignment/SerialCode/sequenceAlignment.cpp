#include<iostream>
#include<iomanip>
#include<cstdio>
using namespace std;

#define MIN(a,b) ( ((a)< (b))?(a):(b) )
#define MAX 8192
#define GE 1
#define GI 2

string X, Y;
int N;
int D[MAX][MAX], I[MAX][MAX], G[MAX][MAX];

int S(int i, int j )
{
if(X[i] != Y[j]) return 1;
return 0;
}

void print(int arr[MAX][MAX], string name)
{
cout<<" =========== "<<name<<" ============ "<<endl;
for(int i=0;i<N;i++, cout<<endl)
	for(int j=0;j<N;j++)
		cout<<setw(5)<<arr[i][j];
return;
}

int main()
{
cin>>N;
cin>>X>>Y;

X=X.substr(0,N);
Y=Y.substr(0,N);

G[0][0] =0;

/* Easy to parallelize */
for(int i=0;i<N;i++)
	{
		if(i>0) {
			G[i][0] =  GI + GE*i;
			G[0][i] =  GI + GE*i;
			}
		I[i][0] = G[i][0] + GE;
		D[0][i] = GE + G[0][i];
	}	


/* Computing i-th row of D could be easily parallelized  */
/* After computing ith row of D, compute vector U  */
/* Compute Vector V from U */

/* Compute vector S = prefix sum of vector V */
/* Compute I using vector S */
/*Compute i-th row of G from U,I */

for(int i=1;i<N;i++)
	for(int j=0;j<N;j++)
		{
			if(i>0 && j>0) D[i][j] = MIN(D[i-1][j] , G[i-1][j] + GI )+GE;
			
			if(i>0 && j>0) I[i][j] = MIN(I[i][j-1], G[i][j-1]+GI )+GE;
			
			if(i>0 && j>0) { 
				int mini  = MIN(D[i][j], I[i][j]);
				int tmp = MIN(mini, G[i-1][j-1]+S(i,j));
				G[i][j] = tmp;
//			printf("Computed G[%d][%d] = %d from D[%d][%d] = %d, I[%d][%d] = %d, G[%d][%d] = %d, S[%d][%d] = %d \n" , i,j, G[i][j], i,j, D[i][j], i,j, I[i][j], i-1, j-1, G[i-1][j-1], i, j, S(i,j)); 
				}
		}

//print(D, "Matrix-D");
//print(I, "Matrix-I");
//print(G, "Matrix-G");

int best = MIN(I[N-1][N-1], G[N-1][N-1]);
best = MIN(best, D[N-1][N-1]);
cout<<"Best alignment cost is "<<best<<endl;
cout<<X<<endl;
cout<<Y<<endl;
return 0;	
}
