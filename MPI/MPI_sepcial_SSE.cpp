#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <string>
#include <mpi.h>
#include <vector>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2
using namespace std;
const int row = 15000;
const int col = 15000;
vector<int> X(row* col); // ��Ԫ�Ӿ���
vector<int> Xo(col); // Xo[i]��ʾ����Ϊi����Ԫ������Ԫ�Ӿ���ĵ�Xo[i]��
vector<int> B(row* col); // ����Ԫ�о���
vector<int> Bo(row); // Bo[i]��ʾ����Ԫ�еĵ�i�е�����ΪBo[i]
int sum1, sum2, sum3; // sum1��ʾ��Ԫ�ӵ�������sum2��ʾʵʱ�ı���Ԫ�е�������sum3��ʾ��ʼ����ı���Ԫ�е�����
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
            X[i * col + j] = 0;
            B[i * col + j] = 0;
        }
    }
}
void copy(int* a, int* b, size_t size) {
    memcpy(a, b, size);
}

// ����Ԫ���ļ�����
void putX() {
    string line;
    int element;
    int xin = 1;
    ifstream infile("X3.txt");
    if (!infile) {
        cout << "�޷����ļ� X.txt" << endl;
        return;
    }
    while (getline(infile, line)) {
        stringstream ss(line);
        while (ss >> element) {
            if (xin) { // ���µ�һ�е�һ������Ϊ���е�����
                Xo[element] = sum1; // ������Ϊtemp��λ����������Xo[temp]��
                xin = 0;
            }
            X[sum1 * col + element] = 1;
            ss.ignore(); // ���Կո�
        }
        xin = 1;
        sum1++;
    }
    infile.close();
}

// ������Ԫ���ļ�����
void putB() {
    string line;
    int element;
    int xin = 1;
    ifstream infile("B3.txt");
    if (!infile) {
        cout << "�޷����ļ� B.txt" << endl;
        return;
    }
    while (getline(infile, line)) {
        stringstream ss(line);
        while (ss >> element) {
            if (xin) {
                Bo[sum2] = element;
                xin = 0;
            }
            B[sum2 * col + element] = 1;
            ss.ignore(); // ���Կո�
        }
        sum2++;
        xin = 1;
    }
    sum3 = sum2;
    infile.close();
}

void print() { // ����������result.txt��
    ofstream outfile("result3.txt");
    outfile << "����õ�����Ԫ���" << endl;
    for (int i = 0; i < sum3; i++) {
        // ���ȡ��ÿһ��
        if (Bo[i] == -1) { // ���ǿ�����������һ��
            continue;
        }
        int z = Bo[i];
        // ȡ��ÿ�е�ÿһ��Ԫ��
        for (z; z >= 0; z--) {
            if (B[i * col + z]) {
                outfile << z << " ";
            }
        }
        outfile << endl;
    }
}

// �����㷨
void elimination(int start, int end) {
    __m128i t0, t1;//���μĴ���
    for (int i = start; i < end; i++) {
        while (Bo[i] != -1 && sum2 > 0) {
            if (Xo[Bo[i]] != -1) {
                int d = Bo[i];
                int	newfirst = -1;
                d = d+(4 - d % 4);//d����Ϊ4����
                for (int j = d - 4; j >= 0; j -= 4)
                {
                	t0 = _mm_loadu_si128((__m128i*)(&B[i*col+j]));
                	t1 = _mm_loadu_si128((__m128i*)(&X[Xo[Bo[i]]*col+j]));
                	t1 = _mm_xor_si128(t0, t1);
                	_mm_storeu_si128((__m128i*)(&B[i*col+j]), t1);//������Ľ�����ȥ
                	if (newfirst == -1)
                	{
                		for (int m = 3; m >= 0; m--)
                		{
                			if ( newfirst == -1)
                			{
                				if (B[i*col+j + m])
                				{
                					newfirst = j+m ;
                				}
                			}
                		}
                	}
                }
                Bo[i] = newfirst;
            }
            else {
                copy(&X[sum1 * col], &B[i * col], col * sizeof(int));
                Xo[Bo[i]] = sum1;
                sum1++;
                sum2--;
                break;
            }
        }
    }
}
    
void parallel_elimination(vector<int>& local_B, vector<int>& local_Bo, int start, int end) {
    __m128i t0, t1;//���μĴ���
    for (int i = start; i < end; i++) {
        while (local_Bo[i] != -1) {
            if (Xo[local_Bo[i]] != -1) {
                int d = local_Bo[i];
                int	newfirst = -1;
                d = d + (4 - d % 4);//d����Ϊ4����
                for (int j = d - 4; j >= 0; j -= 4)
                {
                    t0 = _mm_loadu_si128((__m128i*)(&local_B[i * col + j]));
                    t1 = _mm_loadu_si128((__m128i*)(&X[Xo[local_Bo[i]] * col + j]));
                    t1 = _mm_xor_si128(t0, t1);
                    _mm_storeu_si128((__m128i*)(&local_B[i * col + j]), t1);//������Ľ�����ȥ
                    if (newfirst == -1)
                    {
                        for (int m = 3; m >= 0; m--)
                        {
                            if (newfirst == -1)
                            {
                                if (local_B[i * col + j + m])
                                {
                                    newfirst = j + m;
                                }
                            }
                        }
                    }
                }
                local_Bo[i] = newfirst;
            }
            else {
                copy(&X[sum1 * col], &local_B[i * col], col * sizeof(int));
                Xo[local_Bo[i]] = sum1;
                sum1++;
                break;
            }
        }
    }
}
    
    

int main(int argc, char** argv) {
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    init();

    // ֻ�и����̣�rankΪ0��ִ�ж�����
    if (rank == 0) {
        putX();
        putB();
    }

    // �㲥��Ԫ�Ӻͱ���Ԫ�еľ����Լ���ر���
    MPI_Bcast(Xo.data(), col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum3, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(X.data(), row * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), row * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bo.data(), row, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = (sum3 + size - 1) / size; // ����ȡ����ȡÿ�����̴������������
    int start = rank * rows_per_proc;
    int end = min(start + rows_per_proc, sum3); // ȷ��end������sum3

    // ���ط���
    vector<int> local_B(rows_per_proc * col, 0);
    vector<int> local_Bo(rows_per_proc, -1);
    // ʹ��MPI_Scatter�ַ�����
    MPI_Scatter(B.data(), rows_per_proc * col, MPI_INT, local_B.data(), rows_per_proc * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(Bo.data(), rows_per_proc, MPI_INT, local_Bo.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    parallel_elimination(local_B, local_Bo, 0, end - start);
    // Gather�����������
    MPI_Gather(local_B.data(), rows_per_proc * col, MPI_INT, B.data(), rows_per_proc * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_Bo.data(), rows_per_proc, MPI_INT, Bo.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // ����rankΪsize-1��ʣ�ಿ��
        if (size > 1 && end < sum3) {
            elimination(end, sum3);
        }
        print(); // �����д���ļ�
    }
    if (rank == 1)
    {
        double end_time = MPI_Wtime();
        printf("Total time: %f seconds\n", end_time - start_time);
    }
    // ȷ�����н��̶�����ȷ��ִ�е�����
    printf("Process %d reached MPI_Finalize\n", rank);
    MPI_Finalize();
    return 0;
}