#include<iostream>
#include<iomanip>
using namespace std;

#define MIN(a,b) ( ((a)< (b))?(a):(b) )
#define MAX 1024
#define GE 1
#define GI 2

string X, Y;
int N;
int D[MAX][MAX], I[MAX][MAX], G[MAX][MAX];

int S(int i, int j )
{
if(X[i] == Y[j]) return 1;
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
for(int i=0;i<N;i++)
	{
		if(i>0) {
			G[i][0] =  GI + GE*i;
			G[0][i] =  GI + GE*i;
			}
		I[i][0] = G[i][0] + GE;
		D[0][i] = GE + G[0][i];
	}	

for(int i=1;i<N;i++)
	for(int j=0;j<N;j++)
		{
			D[i][j] = MIN(D[i-1][j] , G[i-1][j] + GI )+GE;
			
			if(j>0) I[i][j] = MIN(I[i][j-1], G[i][j-1]+GI )+GE;
			
			if(i>0 && j>0) { 
				int tmp  = MIN(D[i][j], I[i][j]);
				tmp = MIN(tmp, G[i-1][j-1]+S(i,j));
				G[i][j] = tmp;
				}
		}

print(D, "Matrix-D");
print(I, "Matrix-I");
print(G, "Matrix-G");

int best = MIN(I[N-1][N-1], G[N-1][N-1]);
best = MIN(best, D[N-1][N-1]);
cout<<"Best alignment cost is "<<best<<endl;
cout<<X<<endl;
cout<<Y<<endl;
return 0;	
}
