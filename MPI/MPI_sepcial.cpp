#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <string>
#include <mpi.h>
#include <vector>
using namespace std;
const int row = 15000;
const int col = 15000;
vector<int> X(row* col); // 消元子矩阵
vector<int> Xo(col); // Xo[i]表示首项为i的消元子在消元子矩阵的第Xo[i]行
vector<int> B(row* col); // 被消元行矩阵
vector<int> Bo(row); // Bo[i]表示被消元行的第i行的首项为Bo[i]
int sum1, sum2, sum3; // sum1表示消元子的行数，sum2表示实时的被消元行的行数，sum3表示开始输入的被消元行的行数
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

// 将消元子文件读入
void putX() {
    string line;
    int element;
    int xin = 1;
    ifstream infile("X3.txt");
    if (!infile) {
        cout << "无法打开文件 X.txt" << endl;
        return;
    }
    while (getline(infile, line)) {
        stringstream ss(line);
        while (ss >> element) {
            if (xin) { // 在新的一行第一个数字为该行的首项
                Xo[element] = sum1; // 将首项为temp的位置行数存在Xo[temp]中
                xin = 0;
            }
            X[sum1 * col + element] = 1;
            ss.ignore(); // 忽略空格
        }
        xin = 1;
        sum1++;
    }
    infile.close();
}

// 将被消元行文件读入
void putB() {
    string line;
    int element;
    int xin = 1;
    ifstream infile("B3.txt");
    if (!infile) {
        cout << "无法打开文件 B.txt" << endl;
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
            ss.ignore(); // 忽略空格
        }
        sum2++;
        xin = 1;
    }
    sum3 = sum2;
    infile.close();
}

void print() { // 将结果输出到result.txt中
    ofstream outfile("result3.txt");
    outfile << "计算得到的消元结果" << endl;
    for (int i = 0; i < sum3; i++) {
        // 逐次取到每一行
        if (Bo[i] == -1) { // 若是空行则跳到下一行
            continue;
        }
        int z = Bo[i];
        // 取到每行的每一个元素
        for (z; z >= 0; z--) {
            if (B[i * col + z]) {
                outfile << z << " ";
            }
        }
        outfile << endl;
    }
}

// 串行算法
void elimination(int start, int end) {
    for (int i = start; i < end; i++) {
        while (Bo[i] != -1 && sum2 > 0) {
            if (Xo[Bo[i]] != -1) {
                int d = Bo[i];
                int newfirst = -1;
                for (d; d >= 0; d--) {
                    B[i * col + d] = B[i * col + d] ^ X[Xo[Bo[i]] * col + d];
                    if (newfirst == -1 && B[i * col + d] != 0) {
                        newfirst = d;
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
    for (int i = start; i < end; i++) {
        while (local_Bo[i] != -1) {
            if (Xo[local_Bo[i]] != -1) {
                int d = local_Bo[i];
                int newfirst = -1;
                for (d; d >= 0; d--) {
                    local_B[i * col + d] = local_B[i * col + d] ^ X[Xo[local_Bo[i]] * col + d];
                    if (newfirst == -1 && local_B[i * col + d] != 0) {
                        newfirst = d;
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

    // 只有根进程（rank为0）执行读操作
    if (rank == 0) {
        putX();
        putB();
    }

    // 广播消元子和被消元行的矩阵以及相关变量
    MPI_Bcast(Xo.data(), col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum3, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(X.data(), row * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), row * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bo.data(), row, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = (sum3 + size - 1) / size; // 向上取整获取每个进程处理的数据行数
    int start = rank * rows_per_proc;
    int end = min(start + rows_per_proc, sum3); // 确保end不超过sum3

    // 本地分配
    vector<int> local_B(rows_per_proc * col, 0);
    vector<int> local_Bo(rows_per_proc, -1);
    // 使用MPI_Scatter分发数据
    MPI_Scatter(B.data(), rows_per_proc * col, MPI_INT, local_B.data(), rows_per_proc * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(Bo.data(), rows_per_proc, MPI_INT, local_Bo.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    parallel_elimination(local_B, local_Bo, 0, end - start);
    // Gather结果到根进程
    MPI_Gather(local_B.data(), rows_per_proc * col, MPI_INT, B.data(), rows_per_proc * col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_Bo.data(), rows_per_proc, MPI_INT, Bo.data(), rows_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // 处理rank为size-1的剩余部分
        if (size > 1 && end < sum3) {
            elimination(end, sum3);
        }
        print(); // 将结果写入文件
    }
    double end_time = MPI_Wtime();
    printf("Total time: %f seconds\n", end_time - start_time);
    
    MPI_Finalize();
    return 0;
}