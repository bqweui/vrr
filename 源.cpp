#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;
void comman(int n)
{
	int* sum = new int[n];
	int** b = new int* [n];                                                         
	for (int i = 0; i < n; i++)
	{
		b[i] = new int[n];
	}
	int* a = new int[n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i] = i;
			b[i][j] = i + j;
		}
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER *) & freq);
	QueryPerformanceCounter((LARGE_INTEGER *) & head);
	for (int z = 0; z < 10; z++)
	{
		for (int i = 0; i < n; i++)
		{
			sum[i] = 0;
			for (int j = 0; j < n; j++)
			{
				sum[i] += b[j][i] * a[j];
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER *) & tail);
	for (int i = 0; i < n; i++)
	{
		delete[] b[i];
	}
	delete[]b;
	delete[]sum;
	delete[]a;
	cout << (tail - head) * 1000.0 / freq/10 << endl;
}
void cache(int n) 
{
	int* sum = new int[n];
	int** b = new int* [n];
	for (int i = 0; i < n; i++)
	{
		b[i] = new int[n];
	}
	int* a = new int[n];
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i] = i;
			b[i][j] = i + j;
		}
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int z = 0; z < 10; z++)
	{
		for (int j = 0; j < n; j++)
		{
			sum[j] = 0;
			for (int i = 0; i < n; i++)
			{
				sum[i] += b[j][i] * a[j];
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	for (int i = 0; i < n; i++)
	{
		delete[] b[i];
	}
	delete[]b;
	delete[]sum;
	delete[]a;
	cout << (tail - head) * 1000.0 / freq /10<< endl;
}
 int main()
 {
	 int n=100;
	 for (; n <= 20000;)
	 {
		 comman(n);
		 cache(n);
		 if (n < 1000)
		 {
			 n += 100;
		 }
		 else if (n >= 1000)
		 {
			 n += 1000;
		 }
	 }
	 return 0;
}


