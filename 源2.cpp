#include<iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;
void comman(int n)
{
	int sum = 0;
	int* a = new int[n];
	for (int i = 0; i < n; i++)
	{
		a[i] = i;
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int z = 0; z < 100; z++)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[i];
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	delete[] a;
	cout  << (tail - head) * 1000.0 / freq / 100 << endl;
}
void you(int n)
{
	int sum = 0;
	int* a = new int[n];
	for (int i = 0; i < n; i++)
	{
		a[i] = i;
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int z = 0; z < 100; z++)
	{
		int sum1 = 0;
		int sum2 = 0;
		int sum3 = 0;
		int sum4 = 0;
		int sum5 = 0;
		for (int i = 0; i < n; i += 5)
		{
			sum1 += a[i];
			sum2 += a[i + 1];
			sum3 += a[i + 2];
			sum4 += a[i + 3];
			sum5 += a[i + 4];
		}
		sum = sum1 + sum2 + sum3 + sum4 + sum5;
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	delete[] a;
	cout  << (tail - head) * 1000.0 / freq / 100 << endl;
}
void unroll(int n)
{
	int sum = 0;
	int* a = new int[n];
	for (int i = 0; i < n; i++)
	{
		a[i] = i;
	}
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int z = 0; z < 100; z++)
	{
		for (int i = 0; i < n; i += 5) {
			sum += i;
			sum += i + 1;
			sum += i + 2;
			sum += i + 3;
			sum += i + 4;
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	delete[] a;
	cout  << (tail - head) * 1000.0 / freq / 100 << endl;
}
int main()
{
	int n = 10000;
	for (; n <= 100000;)
	{
	    comman(n);
	    you(n);
		unroll(n);
		n += 10000;
	}
	return 0;
}