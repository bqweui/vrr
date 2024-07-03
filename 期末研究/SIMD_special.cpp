#include <iostream>
#include<fstream>
#include <sstream>
#include<cstring>
#include<time.h>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
#include <string>
//#include <arm_neon.h>//Neon
using namespace std;
const int row = 15000;
const int col = 15000;
int X[row][col];//��Ԫ�Ӿ���
int Xo[col];//Xo[i]��ʾ����Ϊi����Ԫ������Ԫ�Ӿ���ĵ�Xo[i]��
int B[row][col];//����Ԫ�о���
int Bo[row];//Bo[i]��ʾ����Ԫ�еĵ�i�е�����ΪBo[i]
int sum1, sum2, sum3;//sum1��ʾ��Ԫ�ӵ�������sum2��ʾʵʱ�ı���Ԫ�е�������sum3��ʾ��ʼ����ı���Ԫ�е�����
//��ʽ��ֵ����ֹ����
void init()
{
	sum1 = 0;
	sum2 = 0;
	sum3 = 0;
	for (int i = 0; i < col; i++)
	{
		Xo[i] = -1;
	}
	for (int i = 0; i < row; i++)
	{
		Bo[i] = -1;
		for (int j = 0; j < col; j++)
		{
			X[i][j] = 0;
			B[i][j] = 0;
		}
	}
}
void copy(int* a, int* b, size_t size)
{
	memcpy(a, b, size);
}
//����Ԫ���ļ�����
void putX()
{
	string line;
	int element;
	int xin = 1;
	ifstream infile("X7.txt");
	if (!infile)
	{
		cout << "�޷����ļ� X.txt" << endl;
		return;
	}
	while (getline(infile, line))
	{
		stringstream ss(line);
		while (ss >> element)
		{
			if (xin)//���µ�һ�е�һ������Ϊ���е�����
			{
				Xo[element] = sum1;//������Ϊtemp��λ����������Xo[temp]��
				xin = 0;
			}
			X[sum1][element] = 1;
			ss.ignore();//���Կո�
		}
		xin = 1;
		sum1++;
	}
	infile.close();
}
//������Ԫ���ļ�����
void putB()
{
	string line;
	int element;
	int xin = 1;
	ifstream infile("B7.txt");
	if (!infile)
	{
		cout << "�޷����ļ� B.txt" << endl;
		return;
	}
	while (getline(infile, line))
	{
		stringstream ss(line);
		while (ss >> element)
		{
			if (xin)
			{
				Bo[sum2] = element;
				xin = 0;
			}
			B[sum2][element] = 1;
			ss.ignore();//���Կո�
		}
		sum2++;
		xin = 1;
	}
	sum3 = sum2;
	infile.close();
}
//�����㷨
void sequential()
{
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)//�˱���Ԫ�в���ȫ0��ʵʱ�ı���Ԫ�л����ڣ���Ϊ��
		{
			if (Xo[Bo[i]] != -1)//��Ӧ����Ԫ�Ӵ���
			{
				int d = Bo[i] ;
				int newfirst = -1;
				for (d; d >= 0; d--)
				{
					B[i][d] = B[i][d] ^ X[Xo[Bo[i]]][d];//��ÿλ����������㣬�������Ժ�����Ԫ�ؽ���������
					if (newfirst == -1)//�����Ԫ��δ���и�������и���
					{	//�ҵ���һ��Ϊ1��λ��
						if (B[i][d] == 0)
						{
							continue;
						}
						else
						{
							newfirst = d;
						}
					}
				}
				Bo[i] = newfirst;
			}
			else
			{
				//���ñ���Ԫ�е�������뵽��Ԫ�Ӻ󣬽��ñ���Ԫ�е�����λ�õ�Xoֵ��Ϊ���뵽��Ԫ�Ӻ����
				copy(X[sum1], B[i], sizeof(B[i]));//��B[i]����Ϊ��Ԫ��
				Xo[Bo[i]] = sum1;
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
//�����㷨
void parallel_SSE()
{
	__m128i t0, t1;//���μĴ���
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)
		{
			if (Xo[Bo[i]] != -1)//��Ӧ����Ԫ�д���
			{
				int d = Bo[i] ;
				int	newfirst = -1;
				d = d+(4 - d % 4);//d����Ϊ4����
				for (int j = d - 4; j >= 0; j -= 4)
				{
					t0 = _mm_loadu_si128((__m128i*)(B[i] + j));
					t1 = _mm_loadu_si128((__m128i*)(X[Xo[Bo[i]]] + j));
					t1 = _mm_xor_si128(t0, t1);
					_mm_storeu_si128((__m128i*)(B[i] + j), t1);//������Ľ�����ȥ
					if (newfirst == -1)
					{
						for (int m = 3; m >= 0; m--)
						{
							if ( newfirst == -1)
							{
								if (B[i][j + m])
								{
									newfirst = j+m ;
								}
							}
						}
					}
				}
				Bo[i] = newfirst;
			}
			else
			{
				Xo[Bo[i]] = sum1;
				copy(X[sum1], B[i], sizeof(B[i]));//��B[i]����Ϊ��Ԫ��
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
void parallel_AVX()
{
	__m256i t0, t1;//���μĴ���
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)
		{
			if (Xo[Bo[i]] != -1)//��Ӧ����Ԫ�д���
			{
				int d = Bo[i];
				int	newfirst = -1;
				d = d + (8 - d % 8);//d����Ϊ4����
				for (int j = d - 8; j >= 0; j -= 8)
				{
					t0 = _mm256_loadu_si256((__m256i*)(B[i] + j));
					t1 = _mm256_loadu_si256((__m256i*)(X[Xo[Bo[i]]] + j));
					t1 = _mm256_xor_si256(t0, t1);
					_mm256_storeu_si256((__m256i*)(B[i] + j), t1);//������Ľ�����ȥ
					if (newfirst == -1)
					{
						for (int m = 7; m >= 0; m--)
						{
							if (newfirst == -1)
							{
								if (B[i][j + m])
								{
									newfirst = j + m;
								}
							}
						}
					}
				}
				Bo[i] = newfirst;
			}
			else
			{
				Xo[Bo[i]] = sum1;
				copy(X[sum1], B[i], sizeof(B[i]));//��B[i]����Ϊ��Ԫ��
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
void print()
{//����������result.txt��
	ofstream outfile("result7.txt");
	outfile << "����õ�����Ԫ���" << endl;
	for (int i = 0; i < sum3; i++)
	{
		//���ȡ��ÿһ��
		if (Bo[i] == -1)//���ǿ�����������һ��
		{
			continue;
		}
		int z = Bo[i] ;
		//ȡ��ÿ�е�ÿһ��Ԫ��
		for (z; z >= 0; z--)
		{
			if (B[i][z])
			{
				outfile  << z << " ";
			}
		}
		outfile << endl;
	}
}
int main()
{
	/*init();
	putX();
	putB();*/
	long long head, tail, freq;
	/*QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	sequential();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "s" <<" " << (tail - head) * 1000.0 / freq << endl;*/
	/*print();*/
	init();
	putX();
	putB();
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_SSE();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "p"<<" " << (tail - head) * 1000.0 / freq << endl;
	/*print();*/
	/*init();
	putX();
	putB();
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_AVX();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "A" << " " << (tail - head) * 1000.0 / freq << endl;*/
	print();
	return 0;
}