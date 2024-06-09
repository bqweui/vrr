
#include<mpi.h>
#include<stdio.h>
#include<math.h>
#include<cstring>
#include <iostream>
#include <windows.h>
#include<algorithm>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;
const int  Process = 7;
const int n = 1000;
float Gauss[n][n];
void init()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Gauss[i][j] = float(rand()) / 10;
		}
	}
}
void sequential()
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

void print()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << Gauss[i][j] << " ";
		}
		cout << endl;
	}
}
int main(int argc, char* argv[])
{//进程0负责整合结果
	int id;
	MPI_Status status;
	MPI_Init(NULL, NULL);
	double time1, time2;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	int start = id * (n - n % Process) / Process, end = (id == Process - 1) ? n - 1 : id * (n - n % Process) / Process + (n - n % Process) / Process - 1;
	if (id == 0)
	{
		init();
		for (int i = 1; i < Process; ++i)
		{
			int start1 = i * (n - n % Process) / Process, end1 = (i == Process - 1) ? n - 1 : i * (n - n % Process) / Process + (n - n % Process) / Process - 1;
			MPI_Send(&Gauss[start1][0], (end1 - start1 + 1) * n, MPI_FLOAT, i, start1, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&Gauss[start][0], (end - start + 1) * n, MPI_FLOAT, 0, start, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = MPI_Wtime();
	__m256 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
		if (id == 0)
		{
			float base1[8] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
			t1 = _mm256_loadu_ps(base1);
			int j = 0;
			for (j = k + 1; j + 7 < n; j += 8)
			{
				t2 = _mm256_loadu_ps(Gauss[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(Gauss[k] + j, t3);
			}
			//剩下部分的处理
			for (j; j < n; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
			for (int j = 1; j < Process; ++j)
			{
				MPI_Send(&Gauss[k][0], n, MPI_FLOAT, j, k, MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&Gauss[k][0], n, MPI_FLOAT, 0, k, MPI_COMM_WORLD, &status);
		}
		if (end >= k + 1)
		{
			//在当前处理的范围内
			for (int i = max(start, k + 1); i <= end; ++i)
			{
				float base2[8] = { Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k] };
				t1 = _mm256_loadu_ps(base2);
				int j = 0;
				for (j = k + 1; j + 7 < n; j += 8)
				{
					t2 = _mm256_loadu_ps(Gauss[k] + j);
					t3 = _mm256_loadu_ps(Gauss[i] + j);
					t4 = _mm256_mul_ps(t1, t2);
					t3 = _mm256_sub_ps(t3, t4);
					_mm256_storeu_ps(Gauss[i] + j, t3);
				}
				//剩下部分的处理
				for (j; j < n; j++)
				{
					Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
				}
				Gauss[i][k] = 0.0;
				//最后一次处理
				if (i == k + 1 && id != 0)
				{
					MPI_Send(&Gauss[i][0], n, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
				}
			}
		}
		if (id == 0 && k + 1 > end && k + 1 < n)
		{
			MPI_Recv(&Gauss[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, k + 1, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	time2 = MPI_Wtime();
	cout << "MPIAlgorithm time " << " " << id << " " << (time2 - time1) * 1000 << " ms" << endl;
	MPI_Finalize();
	return 0;
}