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
#include <immintrin.h> //AVX、AVX2
#include <string>
//#include <arm_neon.h>//Neon
using namespace std;
const int row = 15000;
const int col = 15000;
int X[row][col];//消元子矩阵
int Xo[col];//Xo[i]表示首项为i的消元子在消元子矩阵的第Xo[i]行
int B[row][col];//被消元行矩阵
int Bo[row];//Bo[i]表示被消元行的第i行的首项为Bo[i]
int sum1, sum2, sum3;//sum1表示消元子的行数，sum2表示实时的被消元行的行数，sum3表示开始输入的被消元行的行数
//显式赋值，防止出错
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
//将消元子文件读入
void putX()
{
	string line;
	int element;
	int xin = 1;
	ifstream infile("X7.txt");
	if (!infile)
	{
		cout << "无法打开文件 X.txt" << endl;
		return;
	}
	while (getline(infile, line))
	{
		stringstream ss(line);
		while (ss >> element)
		{
			if (xin)//在新的一行第一个数字为该行的首项
			{
				Xo[element] = sum1;//将首项为temp的位置行数存在Xo[temp]中
				xin = 0;
			}
			X[sum1][element] = 1;
			ss.ignore();//忽略空格
		}
		xin = 1;
		sum1++;
	}
	infile.close();
}
//将被消元行文件读入
void putB()
{
	string line;
	int element;
	int xin = 1;
	ifstream infile("B7.txt");
	if (!infile)
	{
		cout << "无法打开文件 B.txt" << endl;
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
			ss.ignore();//忽略空格
		}
		sum2++;
		xin = 1;
	}
	sum3 = sum2;
	infile.close();
}
//串行算法
void sequential()
{
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)//此被消元行不是全0且实时的被消元行还存在，不为空
		{
			if (Xo[Bo[i]] != -1)//对应的消元子存在
			{
				int d = Bo[i] ;
				int newfirst = -1;
				for (d; d >= 0; d--)
				{
					B[i][d] = B[i][d] ^ X[Xo[Bo[i]]][d];//将每位进行异或运算，将首项以后所有元素进行异或计算
					if (newfirst == -1)//如果消元后还未进行更新则进行更新
					{	//找到第一个为1的位置
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
				//将该被消元行的首项加入到消元子后，将该被消元行的首项位置的Xo值置为加入到消元子后的行
				copy(X[sum1], B[i], sizeof(B[i]));//将B[i]升级为消元子
				Xo[Bo[i]] = sum1;
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
//并行算法
void parallel_SSE()
{
	__m128i t0, t1;//整形寄存器
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)
		{
			if (Xo[Bo[i]] != -1)//对应的消元行存在
			{
				int d = Bo[i] ;
				int	newfirst = -1;
				d = d+(4 - d % 4);//d设置为4倍数
				for (int j = d - 4; j >= 0; j -= 4)
				{
					t0 = _mm_loadu_si128((__m128i*)(B[i] + j));
					t1 = _mm_loadu_si128((__m128i*)(X[Xo[Bo[i]]] + j));
					t1 = _mm_xor_si128(t0, t1);
					_mm_storeu_si128((__m128i*)(B[i] + j), t1);//将异或后的结果存回去
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
				copy(X[sum1], B[i], sizeof(B[i]));//将B[i]升级为消元子
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
void parallel_AVX()
{
	__m256i t0, t1;//整形寄存器
	for (int i = 0; i < sum3; i++)
	{
		while (Bo[i] != -1 && sum2 > 0)
		{
			if (Xo[Bo[i]] != -1)//对应的消元行存在
			{
				int d = Bo[i];
				int	newfirst = -1;
				d = d + (8 - d % 8);//d设置为4倍数
				for (int j = d - 8; j >= 0; j -= 8)
				{
					t0 = _mm256_loadu_si256((__m256i*)(B[i] + j));
					t1 = _mm256_loadu_si256((__m256i*)(X[Xo[Bo[i]]] + j));
					t1 = _mm256_xor_si256(t0, t1);
					_mm256_storeu_si256((__m256i*)(B[i] + j), t1);//将异或后的结果存回去
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
				copy(X[sum1], B[i], sizeof(B[i]));//将B[i]升级为消元子
				sum1++;
				sum2--;
				break;
			}
		}
	}
}
void print()
{//将结果输出到result.txt中
	ofstream outfile("result7.txt");
	outfile << "计算得到的消元结果" << endl;
	for (int i = 0; i < sum3; i++)
	{
		//逐次取到每一行
		if (Bo[i] == -1)//若是空行则跳到下一行
		{
			continue;
		}
		int z = Bo[i] ;
		//取到每行的每一个元素
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