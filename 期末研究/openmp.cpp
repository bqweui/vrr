#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <omp.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <arm_neon.h>//Neon
using namespace std;
const int thread_num = 5;
void init(float** Gauss,int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Gauss[i][j] = float(rand()) / 10;
		}
	}
}
void print(float** Gauss, int n)
{
	for (int i = 0; i < n; i++)
	{
		for(int j=0;j<n;j++)
		{
			cout << Gauss[i][j] << " ";
		}
		cout << endl;
	}
}
void sequential(float** Gauss, int n)
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
void mp(float** Gauss, int n)
{
#pragma omp parallel if(parallel), num_threads(thread_num), private(i, j, k, tmp)
	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		for (int j = k + 1; j < n; j++)
		{
			float tmp = Gauss[k][k];
			Gauss[k][j] = Gauss[k][j]/ tmp;
		}	
		Gauss[k][k] = 1.0;
		#pragma omp for
		for (int i = k + 1; i < n; i++) 
		{
			float tmp = Gauss[i][k];
			for (int j = k + 1; j < n; j++)
			{
				Gauss[i][j] = Gauss[i][j]- tmp * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}
void mp_SSE(float** Gauss, int n)
{
#pragma omp parallel if(parallel), num_threads(thread_num), private(i, j, k, t1,t2,t3,t4,base1,base2)
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
#pragma omp single
		float base1[4] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
		t1 = _mm_loadu_ps(base1);
		int j = 0;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			t2 = _mm_loadu_ps(Gauss[k] + j);
			t3 = _mm_div_ps(t2, t1);
			_mm_storeu_ps(Gauss[k] + j, t3);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j]/Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
#pragma omp for
		for (int i = k + 1; i < n; i++)
		{
			float base2[4] = { Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k] };
			t1 = _mm_loadu_ps(base2);
			int j = 0;
			for (j = k + 1; j + 3 < n; j += 4)
			{
				t2 = _mm_loadu_ps(Gauss[k] + j);
				t3 = _mm_loadu_ps(Gauss[i] + j);
				t4 = _mm_mul_ps(t1, t2);
				t3 = _mm_sub_ps(t3, t4);
				_mm_storeu_ps(Gauss[i] + j, t3);
			}
			//剩下部分的处理
			for (j; j < n; j++)
			{
				Gauss[i][j] = Gauss[i][j]-Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
	}
}
void mp_AVX(float** Gauss, int n)
{
#pragma omp parallel if(parallel), num_threads(thread_num), private(i, j, k, t1,t2,t3,t4,base1,base2)
	__m256 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
#pragma omp single
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
#pragma omp for
		for (int i = k + 1; i < n; i++)
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
		}
	}
}
void mp_Neon(float** Gauss, int n)
{
#pragma omp parallel if(parallel), num_threads(thread_num), private(i, j, k, t1,t2,t3,t4)
	float32x4_t t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
#pragma omp single
		t1 = vld1q_dup_f32(Gauss[k] + k);
		int j = 0;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			t2 = vld1q_f32(Gauss[k] + j);
			t1 = vrecpeq_f32(t1);
			t3 = vmulq_f32(t2, t1);
			vst1q_f32(Gauss[k] + j, t3);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
#pragma omp for
		for (int i = k + 1; i < n; i++)
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
			//剩下部分的处理
			for (j; j < n; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
	}
}
int main()
{
	int n;
	cin >> n;
	float** Gauss = new float* [n + 1];
	for (int i = 0; i < n + 1; i++)
	{
		Gauss[i] = new float[n + 1];
	}
	long long head, tail, freq;
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sequential(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout <<"s"<<" " << (tail - head) * 1000.0 / freq << endl;

	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	mp(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "mp" <<" " << (tail - head) * 1000.0 / freq << endl;

	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	mp_SSE(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "mp_SSE" <<" " << (tail - head) * 1000.0 / freq << endl;

	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	mp_AVX(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "mp_AVX" <<" " << (tail - head) * 1000.0 / freq << endl;

	init(Gauss, n);
	auto start_time_neon3 = std::chrono::high_resolution_clock::now();
	mp_Neon(Gauss, n);
	auto end_time_neon3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon3 = end_time_neon3 - start_time_neon3;
	std::cout << "mp_NEON elimination time for size " << n << ": " << total_time_neon3.count() << " milliseconds\n";
	print(Gauss, n);
	return 0;
}