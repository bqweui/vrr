#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include<Windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include<pthread.h>
#include <semaphore.h>
#include <arm_neon.h>//Neon
using namespace std;
typedef struct
{
	int	threadId;
	float** Gauss;
	int n;
} threadParm_t;
typedef struct {
	 int k; //消去的轮次
	 int threadId; // 线程 id
	 float** Gauss;
	 int n;
}threadParam_t;
const int thread_num = 3;
sem_t sem_main;
sem_t sem_workerstart[thread_num]; // 每个线程有自己专属的信号量
sem_t sem_workerend[thread_num];
sem_t sem_leader;
sem_t sem_Divsion[thread_num-1];
sem_t sem_Elimination[thread_num - 1];
pthread_barrier_t barrier1;
pthread_barrier_t barrier2;
pthread_t threads[thread_num];
threadParm_t threadParm[thread_num];
void init(float** Gauss, int n)
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
		for (int j = 0; j < n; j++)
		{
			std::cout << Gauss[i][j] << " ";
		}
		std::cout << endl;
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
void* pthread_barrier(void* parm)
{
	threadParm_t* p = (threadParm_t*)parm;
	int N = p->n;
	float** Gauss = p->Gauss;
	int id = p->threadId;
	for (int k = 0; k < N; k++)
	{
		if (id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1 + id ; i < N; i+= thread_num)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
		pthread_barrier_wait(&barrier2);
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_barrier_SSE(void* parm)
{
	__m128 t1, t2, t3, t4;
	threadParm_t* p = (threadParm_t*)parm;
	int N = p->n;
	float** Gauss = p->Gauss;
	int id = p->threadId;
	for (int k = 0; k < N; k++)
	{
		if (id == 0)
		{
			float base1[4] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
			t1 = _mm_loadu_ps(base1);
			int j = 0;
			for (j = k + 1; j + 3 < N; j += 4)
			{
				t2 = _mm_loadu_ps(Gauss[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(Gauss[k] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j]/ Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1 + id; i < N; i += thread_num)
		{
			float base2[4] = { Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k] };
			t1 = _mm_loadu_ps(base2);
			int j = 0;
			for (j = k + 1; j + 3 < N; j += 4)
			{
				t2 = _mm_loadu_ps(Gauss[k] + j);
				t3 = _mm_loadu_ps(Gauss[i] + j);
				t4 = _mm_mul_ps(t1, t2);
				t3 = _mm_sub_ps(t3, t4);
				_mm_storeu_ps(Gauss[i] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j]-Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		pthread_barrier_wait(&barrier2);
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_barrier_AVX(void* parm)
{
	__m256 t1, t2, t3, t4;
	threadParm_t* p = (threadParm_t*)parm;
	int N = p->n;
	float** Gauss = p->Gauss;
	int id = p->threadId;
	for (int k = 0; k < N; k++)
	{
		if (id == 0)
		{
			float base1[8] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
			t1 = _mm256_loadu_ps(base1);
			int j = 0;
			for (j = k + 1; j + 7 < N; j += 8)
			{
				t2 = _mm256_loadu_ps(Gauss[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(Gauss[k] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1 + id; i < N; i += thread_num)
		{
			float base2[8] = { Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k],Gauss[i][k] };
			t1 = _mm256_loadu_ps(base2);
			int j = 0;
			for (j = k + 1; j + 7 < N; j += 8)
			{
				t2 = _mm256_loadu_ps(Gauss[k] + j);
				t3 = _mm256_loadu_ps(Gauss[i] + j);
				t4 = _mm256_mul_ps(t1, t2);
				t3 = _mm256_sub_ps(t3, t4);
				_mm256_storeu_ps(Gauss[i] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		pthread_barrier_wait(&barrier2);
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_barrier_Neon(void* parm)
{
	float32x4_t t1, t2, t3, t4;
	threadParm_t* p = (threadParm_t*)parm;
	int N = p->n;
	float** Gauss = p->Gauss;
	int id = p->threadId;
	for (int k = 0; k < N; k++)
	{
		if (id == 0)
		{
			t1 = vld1q_dup_f32(Gauss[k] + k);
			int j = 0;
			for (j = k + 1; j + 3 < N; j += 4)
			{
				t2 = vld1q_f32(Gauss[k] + j);
				t1 = vrecpeq_f32(t1);
				t3 = vmulq_f32(t2, t1);
				vst1q_f32(Gauss[k] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1 + id; i < N; i += thread_num)
		{
			t1 = vld1q_dup_f32(Gauss[i] + k);
			int j = 0;
			for (j = k + 1; j + 3 < N; j += 4)
			{
				t2 = vld1q_f32(Gauss[k] + j);
				t3 = vld1q_f32(Gauss[i] + j);
				t4 = vmulq_f32(t1, t2);
				t3 = vsubq_f32(t3, t4);
				vst1q_f32(Gauss[i] + j, t3);
			}
			//剩下部分的处理
			for (j; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		pthread_barrier_wait(&barrier2);
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_dynamic(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int k = p-> k; //消去的轮次
	int id = p->threadId; //线程编号
	int i = k + id + 1; //获取自己的计算任务
	int n = p->n;
	float** Gauss = p->Gauss;
	for(int j = k + 1; j < n; ++j)
	{
		Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
	} 
	if (i < n)
	{
		Gauss[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_dynamic_SSE(void* parm)
{
	__m128 t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int k = p->k; //消去的轮次
	int id = p->threadId; //线程编号
	int i = k + id + 1; //获取自己的计算任务
	int n = p->n;
	float** Gauss = p->Gauss;
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
	if (i < n)
	{
		Gauss[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_dynamic_AVX(void* parm)
{
	__m256 t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int k = p->k; //消去的轮次
	int id = p->threadId; //线程编号
	int i = k + id + 1; //获取自己的计算任务
	int n = p->n;
	float** Gauss = p->Gauss;
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
	if (i < n)
	{
		Gauss[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_dynamic_Neon(void* parm)
{
	float32x4_t t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int k = p->k; //消去的轮次
	int id = p->threadId; //线程编号
	int i = k + id + 1; //获取自己的计算任务
	int n = p->n;
	float** Gauss = p->Gauss;
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
	if (i < n)
	{
		Gauss[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem1(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p ->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	for (int k = 0; k < n; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		for (int i = k + 1 + t_id; i < n; i += thread_num)//消去
		{
			for (int j = k + 1; j < n; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem1_SSE(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < n; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		for (int i = k + 1 + t_id; i < n; i += thread_num)//消去
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
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem1_AVX(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	__m256 t1, t2, t3, t4;
	for (int k = 0; k < n; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		for (int i = k + 1 + t_id; i < n; i += thread_num)//消去
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
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem1_Neon(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	float32x4_t t1, t2, t3, t4;
	for (int k = 0; k < n; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		for (int i = k + 1 + t_id; i < n; i += thread_num)//消去
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
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem2(void* parm)
{
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p ->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id-1]); // 阻塞，等待完成除法操作
		}
		 //t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num-1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}
		for (int i = k + 1 + t_id; i < n; i += thread_num)
		{
			for (int j = k + 1; j < n; ++j)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0.0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num-1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}

			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id-1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem2_SSE(void* parm)
{
	__m128 t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
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
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}
		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}
		for (int i = k + 1 + t_id; i < n; i += thread_num)
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
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}

			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem2_AVX(void* parm)
{
	__m256 t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
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
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}
		 //t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}
		for (int i = k + 1 + t_id; i < n; i += thread_num)
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
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}

			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return NULL;
}
void* pthread_sem2_Neon(void* parm)
{
	float32x4_t t1, t2, t3, t4;
	threadParam_t* p = (threadParam_t*)parm;
	int t_id = p->threadId;
	int n = p->n;
	float** Gauss = p->Gauss;
	for (int k = 0; k < n; ++k)
	{
		if (t_id == 0)
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
			//剩下部分的处理
			for (j; j < n; j++)
			{
				Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
			}
			Gauss[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}
		 //t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}
		for (int i = k + 1 + t_id; i < n; i += thread_num)
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
		if (t_id == 0)
		{
			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}

			for (int i = 0; i < thread_num - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return NULL;
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
	//1
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sequential(Gauss, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s" <<" " << (tail - head) * 1000.0 / freq << endl;
	//2
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	pthread_barrier_init(&barrier1, NULL, thread_num);
	pthread_barrier_init(&barrier2, NULL, thread_num);
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		threadParm[i].Gauss = Gauss;
		threadParm[i].n = n;
		pthread_create(&threads[i], NULL, pthread_barrier, (void*)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		pthread_join(threads[i], NULL);
	}	
	pthread_barrier_destroy(&barrier1);
	pthread_barrier_destroy(&barrier2);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "barr" << " " << (tail - head) * 1000.0 / freq << endl;
	//3
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	pthread_barrier_init(&barrier1, NULL, thread_num);
	pthread_barrier_init(&barrier2, NULL, thread_num);
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		threadParm[i].Gauss = Gauss;
		threadParm[i].n = n;
		pthread_create(&threads[i], NULL, pthread_barrier_SSE, (void*)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	pthread_barrier_destroy(&barrier1);
	pthread_barrier_destroy(&barrier2);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "barr_SSE" << " " << (tail - head) * 1000.0 / freq << endl;
	//4
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	pthread_barrier_init(&barrier1, NULL, thread_num);
	pthread_barrier_init(&barrier2, NULL, thread_num);
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		threadParm[i].Gauss = Gauss;
		threadParm[i].n = n;
		pthread_create(&threads[i], NULL, pthread_barrier_AVX, (void*)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	pthread_barrier_destroy(&barrier1);
	pthread_barrier_destroy(&barrier2);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "barr_AVX" << " " << (tail - head) * 1000.0 / freq << endl;
	//5
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	__m128 t1, t2, t3;
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
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
		//创建工作线程，进行消去操作
		int worker_count = n-1-k; //工作线程数量
		pthread_t* handles1 = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// 创建对应的 Handle
		threadParam_t* param1 = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param1[t_id].k = k;
			param1[t_id].threadId = t_id;
			param1[t_id].n = n;
			param1[t_id].Gauss = Gauss;
		}
			//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_create(&handles1[t_id], NULL, pthread_dynamic_SSE, (void*)&param1[t_id]);
		}
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_join(handles1[t_id], NULL);
		}
		std::free(handles1);
		std::free(param1);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "dy_SSE" << " " << (tail - head) * 1000.0 / freq << endl;
	//6
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	__m256 t11, t21, t31;
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		float base1[8] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
		t11 = _mm256_loadu_ps(base1);
		int j = 0;
		for (j = k + 1; j + 7 < n; j += 8)
		{
			t21 = _mm256_loadu_ps(Gauss[k] + j);
			t31 = _mm256_div_ps(t21, t11);
			_mm256_storeu_ps(Gauss[k] + j, t31);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = n - 1 - k; //工作线程数量
		pthread_t* handles1 = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// 创建对应的 Handle
		threadParam_t* param1 = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param1[t_id].k = k;
			param1[t_id].threadId = t_id;
			param1[t_id].n = n;
			param1[t_id].Gauss = Gauss;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_create(&handles1[t_id], NULL, pthread_dynamic_AVX, (void*)&param1[t_id]);
		}
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_join(handles1[t_id], NULL);
		}
		std::free(handles1);
		std::free(param1);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "dy_AVX" << " " << (tail - head) * 1000.0 / freq << endl;
	//7
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = n - 1 - k; //工作线程数量
		pthread_t* handles1 = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// 创建对应的 Handle
		threadParam_t* param1 = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param1[t_id].k = k;
			param1[t_id].threadId = t_id;
			param1[t_id].n = n;
			param1[t_id].Gauss = Gauss;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_create(&handles1[t_id], NULL, pthread_dynamic, (void*)&param1[t_id]);
		}
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_join(handles1[t_id], NULL);
		}
		std::free(handles1);
		std::free(param1);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "dy" << " " << (tail - head) * 1000.0 / freq << endl;
	//8
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < thread_num; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	pthread_t handles2[thread_num];// 创建对应的 Handle
	threadParam_t param2[thread_num];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param2[t_id].threadId = t_id;
		param2[t_id].n = n;
		param2[t_id].Gauss = Gauss;
		pthread_create(&handles2[t_id], NULL, pthread_sem1, (void*)&param2[t_id]);
	}
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}
			//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_wait(&sem_main);
		}
		
		 //主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles2[t_id], NULL);
	}
	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerstart[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerend[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s1e" << " " << (tail - head) * 1000.0 / freq << endl;
	//9
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < thread_num; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param2[t_id].threadId = t_id;
		param2[t_id].n = n;
		param2[t_id].Gauss = Gauss;
		pthread_create(&handles2[t_id], NULL, pthread_sem1_SSE, (void*)&param2[t_id]);
	}
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
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

		//开始唤醒工作线程
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_wait(&sem_main);
		}

		 //主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles2[t_id], NULL);
	}
	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerstart[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerend[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s1e_SSE" << " " << (tail - head) * 1000.0 / freq << endl;
	//10
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < thread_num; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param2[t_id].threadId = t_id;
		param2[t_id].n = n;
		param2[t_id].Gauss = Gauss;
		pthread_create(&handles2[t_id], NULL, pthread_sem1_AVX, (void*)&param2[t_id]);
	}
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		float base1[8] = { Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k],Gauss[k][k] };
		t11 = _mm256_loadu_ps(base1);
		int j = 0;
		for (j = k + 1; j + 7 < n; j += 8)
		{
			t21 = _mm256_loadu_ps(Gauss[k] + j);
			t31 = _mm256_div_ps(t21, t11);
			_mm256_storeu_ps(Gauss[k] + j, t31);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_wait(&sem_main);
		}

		 //主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles2[t_id], NULL);
	}
	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerstart[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerend[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s1e_AVX" << " " << (tail - head) * 1000.0 / freq << endl;
	//11
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < thread_num - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t handles3[thread_num];// 创建对应的 Handle
	threadParam_t param3[thread_num];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param3[t_id].threadId = t_id;
		param3[t_id].n = n;
		param3[t_id].Gauss = Gauss;
		pthread_create(&handles3[t_id], NULL, pthread_sem2, (void*)&param3[t_id]);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles3[t_id], NULL);
	}
	sem_destroy(&sem_leader);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Divsion[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Elimination[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s2e" << " " << (tail - head) * 1000.0 / freq << endl;
	//12
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < thread_num - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param3[t_id].threadId = t_id;
		param3[t_id].n = n;
		param3[t_id].Gauss = Gauss;
		pthread_create(&handles3[t_id], NULL, pthread_sem2_SSE, (void*)&param3[t_id]);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles3[t_id], NULL);
	}
	sem_destroy(&sem_leader);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Divsion[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Elimination[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s2e_SSE" << " " << (tail - head) * 1000.0 / freq << endl;
	//13
	init(Gauss, n);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < thread_num - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param3[t_id].threadId = t_id;
		param3[t_id].n = n;
		param3[t_id].Gauss = Gauss;
		pthread_create(&handles3[t_id], NULL, pthread_sem2_AVX, (void*)&param3[t_id]);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles3[t_id], NULL);
	}
	sem_destroy(&sem_leader);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Divsion[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Elimination[i]);
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	std::cout << "s2e_AVX" << " " << (tail - head) * 1000.0 / freq << endl;
	print(Gauss, n);
	//14
	init(Gauss, n);
	auto start_time_neon = std::chrono::high_resolution_clock::now();
	pthread_barrier_init(&barrier1, NULL, thread_num);
	pthread_barrier_init(&barrier2, NULL, thread_num);
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		threadParm[i].Gauss = Gauss;
		threadParm[i].n = n;
		pthread_create(&threads[i], NULL, pthread_barrier_Neon, (void*)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	pthread_barrier_destroy(&barrier1);
	pthread_barrier_destroy(&barrier2);
	auto end_time_neon = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon = end_time_neon - start_time_neon;
	std::cout << "barr_NEON elimination time for size " << n << ": " << total_time_neon.count() << " milliseconds\n";
	//15
	init(Gauss, n);
	auto start_time_neon1 = std::chrono::high_resolution_clock::now();
	float32x4_t t12, t22, t32;
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		t12 = vld1q_dup_f32(Gauss[k] + k);
		int j = 0;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			t22 = vld1q_f32(Gauss[k] + j);
			t12 = vrecpeq_f32(t12);
			t32 = vmulq_f32(t22, t12);
			vst1q_f32(Gauss[k] + j, t32);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = n-1-k; //工作线程数量
		pthread_t* handles1 = (pthread_t*)malloc(worker_count * sizeof(pthread_t));// 创建对应的 Handle
		threadParam_t* param1 = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param1[t_id].k = k;
			param1[t_id].threadId = t_id;
			param1[t_id].n = n;
			param1[t_id].Gauss = Gauss;
		}
			//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_create(&handles1[t_id], NULL, pthread_dynamic_Neon, (void*)&param1[t_id]);
		}
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_join(handles1[t_id], NULL);
		}
		std::free(handles1);
		std::free(param1);
	}
	auto end_time_neon1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon1 = end_time_neon1 - start_time_neon1;
	std::cout << "dy_NEON elimination time for size " << n << ": " << total_time_neon1.count() << " milliseconds\n";
	//16
	init(Gauss, n);
	auto start_time_neon2 = std::chrono::high_resolution_clock::now();
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < thread_num; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param2[t_id].threadId = t_id;
		param2[t_id].n = n;
		param2[t_id].Gauss = Gauss;
		pthread_create(&handles2[t_id], NULL, pthread_sem1_Neon, (void*)&param2[t_id]);
	}
	for (int k = 0; k < n; ++k)
	{
		//主线程做除法操作
		t12 = vld1q_dup_f32(Gauss[k] + k);
		int j = 0;
		for (j = k + 1; j + 3 < n; j += 4)
		{
			t22 = vld1q_f32(Gauss[k] + j);
			t12 = vrecpeq_f32(t12);
			t32 = vmulq_f32(t22, t12);
			vst1q_f32(Gauss[k] + j, t32);
		}
		//剩下部分的处理
		for (j; j < n; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < thread_num; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles2[t_id], NULL);
	}
	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerstart[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_workerend[i]);
	}
	auto end_time_neon2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon2 = end_time_neon2 - start_time_neon2;
	std::cout << "s1e_NEON elimination time for size " << n << ": " << total_time_neon2.count() << " milliseconds\n";
	//17
	init(Gauss, n);
	auto start_time_neon3 = std::chrono::high_resolution_clock::now();
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < thread_num - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		param3[t_id].threadId = t_id;
		param3[t_id].n = n;
		param3[t_id].Gauss = Gauss;
		pthread_create(&handles3[t_id], NULL, pthread_sem2_Neon, (void*)&param3[t_id]);
	}
	for (int t_id = 0; t_id < thread_num; t_id++)
	{
		pthread_join(handles3[t_id], NULL);
	}
	sem_destroy(&sem_leader);
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Divsion[i]);
	}
	for (int i = 0; i < thread_num; i++)
	{
		sem_destroy(&sem_Elimination[i]);
	}
	auto end_time_neon3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> total_time_neon3 = end_time_neon3 - start_time_neon3;
	std::cout << "s2e_NEON elimination time for size " << n << ": " << total_time_neon3.count() << " milliseconds\n";
	return 0;

}