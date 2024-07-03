#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <arm_neon.h>//Neon
using namespace std;
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
			Gauss[k][j] = Gauss[k][j]/Gauss[k][k];
		}	
		Gauss[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) 
		{
			for (int j = k + 1; j < n; j++)
			{
				Gauss[i][j] = Gauss[i][j]-Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}
void not_aligned_parallel_SSE_vector(float** Gauss, int n)
{
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
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
void aligned_parallel_SSE_vector(float** Gauss, int n)
{
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
		float base1[4] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
		t1 = _mm_loadu_ps(base1);
		int j = 0;
		for (j = k + 1; j < n; j++) 
		{
			if (((size_t)(Gauss[k] + j)) % 16 == 0)
			{
				break;
			}
			else
			{
				Gauss[k][j] = Gauss[k][j]/Gauss[k][k];
			}
		}
		for (j ; j + 3 < n; j += 4)
		{
			t2 = _mm_load_ps(Gauss[k] + j);
			t3 = _mm_div_ps(t2, t1);
			_mm_store_ps(Gauss[k] + j, t3);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			float base2[4] = { Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k] };
			t1 = _mm_loadu_ps(base2);
			int j = 0;
			for (j = k + 1; j < n; j++)
			{
				if (((size_t)(Gauss[i] + j)) % 16 == 0)
				{
					break;
				}
				else
				{
					Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
				}
			}
			for (j ; j + 3 < n; j += 4)
			{
				t2 = _mm_load_ps(Gauss[k] + j);
				t3 = _mm_load_ps(Gauss[i] + j);
				t4 = _mm_mul_ps(t1, t2);
				t3 = _mm_sub_ps(t3, t4);
				_mm_store_ps(Gauss[i] + j, t3);
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
void parallel_SSE_46_vector(float** Gauss, int n)
{
	__m128 t1, t2, t3;
	for (int k = 0; k < n; k++)
	{
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
void parallel_SSE_813_vector(float** Gauss, int n)
{
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
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
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
	}
}
void parallel_AVX_vector(float** Gauss, int n)
{
	__m256 t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
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
void parallel_Neon_vector(float** Gauss, int n)
{
	float32x4_t t1, t2, t3, t4;
	for (int k = 0; k < n; k++)
	{
		t1 = vld1q_dup_f32(Gauss[k]+k);
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
		for (int i = k + 1; i < n; i++)
		{
			t1 = vld1q_dup_f32(Gauss[i]+k);
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
	init(Gauss, n);
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sequential(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout <<"o" << (tail - head) * 1000.0 / freq << endl;
	//2
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	not_aligned_parallel_SSE_vector(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout<<"s" << (tail - head) * 1000.0 / freq << endl;
	//3
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	aligned_parallel_SSE_vector(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "as" << (tail - head) * 1000.0 / freq << endl;
	//4
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_SSE_46_vector(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "46s" << (tail - head) * 1000.0 / freq << endl;
	//5
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_SSE_813_vector(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "813s" << (tail - head) * 1000.0 / freq << endl;
	//6
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_AVX_vector(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "a" << (tail - head) * 1000.0 / freq << endl;

	/*auto start_time_neon = std::chrono::high_resolution_clock::now();
	parallel_Neon_vector(Gauss, n);
	auto end_time_neon = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon = end_time_neon - start_time_neon;
	std::cout << "NEON elimination time for size " << n << ": " << total_time_neon.count() << " milliseconds\n";*/
	return 0;
}

