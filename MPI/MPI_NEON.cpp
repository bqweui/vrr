
#include<mpi.h>
#include<stdio.h>
#include<math.h>
#include<cstring>
#include <iostream>
#include<algorithm>
#include <arm_neon.h>//Neon
using namespace std;
const int  Process = 10;
const int n = 50;
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
{//����0�������Ͻ��
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
	float32x4_t t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
		if (id == 0)
		{
			t1 = vld1q_dup_f32(Gauss[k] + k);
			int j = 0;
			for (j = k + 1; j + 3 < n; j += 4)
			{
				t2 = vld1q_f32(Gauss[k] + j);
				t1 = vrecpeq_f32(t1);
				t3 = vmulq_f32(t2, t1);
				vst1q_f32(Gauss[k] + j, t3);
			}
			//ʣ�²��ֵĴ���
			for (j; j < n; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
			for (int j = 1; j < Process; ++j)
			{
				MPI_Send(&Gauss[k][0], n, MPI_FLOAT, j, k , MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&Gauss[k][0], n, MPI_FLOAT, 0, k, MPI_COMM_WORLD, &status);
		}
		if (end >= k + 1)
		{
			//�ڵ�ǰ����ķ�Χ��
			for (int i = max(start, k + 1); i <= end; ++i)
			{
				t1 = vld1q_dup_f32(Gauss[i] + k);
				int j = 0;
				for (j = k + 1; j + 3 < n; j += 4)
				{
					t2 = vld1q_f32(Gauss[k] + j);
					t3 = vld1q_f32(Gauss[i] + j);
					t4 = vmulq_f32(t1, t2);
					t3 = vsubq_f32(t3, t4);
					vst1q_f32(Gauss[i] + j, t3);
				}
				//ʣ�²��ֵĴ���
				for (j; j < n; j++)
				{
					Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
				}
				Gauss[i][k] = 0.0;
				//���һ�δ���
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