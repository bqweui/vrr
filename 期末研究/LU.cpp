#include <iostream>
#include <vector>
#include <immintrin.h> 
#include <windows.h>
#include <omp.h>
#include <pthread.h>
using namespace std;
pthread_barrier_t barrier;
void LU(vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, int n) {
    for (int i = 0; i < n; i++)
    {
        // 计算上三角矩阵 U 的第 i 行
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }

        // 计算下三角矩阵 L 的第 i 列
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1; // 对角线元素设为 1
            }
            else {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += (L[k][j] * U[j][i]);
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

void LU_SIMD(vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, int n) {
    for (int i = 0; i < n; i++)
    {
        // 计算上三角矩阵 U 的第 i 行
        for (int k = i; k < n; k++) {
            __m256d sum = _mm256_setzero_pd();
            int j;
            for (j = 0; j <= i - 4; j += 4) {  // 使用SIMD指令处理4的倍数部分
                __m256d l = _mm256_loadu_pd(&L[i][j]);
                __m256d u = _mm256_loadu_pd(&U[j][k]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(l, u));
            }
            double result[4];
            _mm256_storeu_pd(result, sum);
            double sum_scalar = result[0] + result[1] + result[2] + result[3];
            // 处理剩余的元素
            for (; j < i; j++) {
                sum_scalar += L[i][j] * U[j][k];
            }

            U[i][k] = A[i][k] - sum_scalar;
        }

        // 计算下三角矩阵 L 的第 i 列
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1; // 对角线元素设为 1
            }
            else {
                __m256d sum = _mm256_setzero_pd();
                int j;
                for (j = 0; j <= i - 4; j += 4) {  // 使用SIMD指令处理4的倍数部分
                    __m256d l = _mm256_loadu_pd(&L[k][j]);
                    __m256d u = _mm256_loadu_pd(&U[j][i]);
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(l, u));
                }

                double result[4];
                _mm256_storeu_pd(result, sum);
                double sum_scalar = result[0] + result[1] + result[2] + result[3];

                // 处理剩余的元素
                for (; j < i; j++) {
                    sum_scalar += L[k][j] * U[j][i];
                }

                L[k][i] = (A[k][i] - sum_scalar) / U[i][i];
            }
        }
    }
}


void LU_OpenMP(vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
#pragma omp parallel for
        // 计算上三角矩阵 U 的第 i 行
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }
#pragma omp parallel for
        // 计算下三角矩阵 L 的第 i 列
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1; // 对角线元素设为 1
            }
            else {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += (L[k][j] * U[j][i]);
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

void LU_SIMD_OpenMP(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U, int n) {
    // 初始化 L 和 U 矩阵
    L = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    U = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // 计算上三角矩阵 U 的第 i 行
#pragma omp parallel for
        for (int k = i; k < n; k++) {
            __m256d sum = _mm256_setzero_pd();
            int j;
            for (j = 0; j <= i - 4; j += 4) {
                __m256d l = _mm256_loadu_pd(&L[i][j]);
                __m256d u = _mm256_loadu_pd(&U[j][k]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(l, u));
            }
            double result[4];
            _mm256_storeu_pd(result, sum);
            double sum_scalar = result[0] + result[1] + result[2] + result[3];
            // 处理剩余的元素
            for (; j < i; j++) {
                sum_scalar += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum_scalar;
        }
        // 计算下三角矩阵 L 的第 i 列
#pragma omp parallel for
        for (int k = i; k < n; k++) {
            if (i == k) {
                L[i][i] = 1; // 对角线元素设为 1
            }
            else {
                __m256d sum = _mm256_setzero_pd();
                int j;
                for (j = 0; j <= i - 4; j += 4) {
                    __m256d l = _mm256_loadu_pd(&L[k][j]);
                    __m256d u = _mm256_loadu_pd(&U[j][i]);
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(l, u));
                }
                double result[4];
                _mm256_storeu_pd(result, sum);
                double sum_scalar = result[0] + result[1] + result[2] + result[3];
                // 处理剩余的元素
                for (; j < i; j++) {
                    sum_scalar += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum_scalar) / U[i][i];
            }
        }
    }
}



void Verify(vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, int n) {
    vector<vector<double>> LU(n, vector<double>(n, 0));
    // 计算 L 和 U 的乘积
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    // 输出 L * U 的结果
    cout << "L * U 矩阵:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << LU[i][j] << " ";
        }
        cout << endl;
    }

}

void ProcessMatrix_SO(vector<vector<double>>& A, int n, const string& name) {
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_SIMD_OpenMP(A, L, U, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    /*cout << "矩阵 " << name << ":" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }

    cout << "L 矩阵:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << L[i][j] << " ";
        }
        cout << endl;
    }

    cout << "U 矩阵:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << U[i][j] << " ";
        }
        cout << endl;
    }

    Verify(A, L, U, n);*/
    cout << "SO" << " " << (tail - head) * 1000.0 / freq << endl;
}
vector<vector<double>> init(int n) {
    srand(time(0));
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand() % 100 + 1; // 生成 1 到 100 范围内的随机数
        }
    }
    return matrix;
}


void ProcessMatrix(vector<vector<double>>& A, int n, const string& name) {
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU(A, L, U, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

   /* cout << "矩阵 " << name << ":" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }

    cout << "L 矩阵:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << L[i][j] << " ";
        }
        cout << endl;
    }

    cout << "U 矩阵:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << U[i][j] << " ";
        }
        cout << endl;
    }

    Verify(A, L, U, n);*/
    cout << "LU" << " " << (tail - head) * 1000.0 / freq << endl;
}
void ProcessMatrix_OP(vector<vector<double>>& A, int n, const string& name) {
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_OpenMP(A, L, U, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

     /*cout << "矩阵 " << name << ":" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << A[i][j] << " ";
         }
         cout << endl;
     }

     cout << "L 矩阵:" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << L[i][j] << " ";
         }
         cout << endl;
     }

     cout << "U 矩阵:" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << U[i][j] << " ";
         }
         cout << endl;
     }

     Verify(A, L, U, n);*/
    cout << "OP" << " " << (tail - head) * 1000.0 / freq << endl;
}
void ProcessMatrix_SIMD(vector<vector<double>>& A, int n, const string& name) {
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU_SIMD(A, L, U, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

     /*cout << "矩阵 " << name << ":" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << A[i][j] << " ";
         }
         cout << endl;
     }

     cout << "L 矩阵:" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << L[i][j] << " ";
         }
         cout << endl;
     }

     cout << "U 矩阵:" << endl;
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             cout << U[i][j] << " ";
         }
         cout << endl;
     }

     Verify(A, L, U, n);*/
    cout << "S" << " " << (tail - head) * 1000.0 / freq << endl;
}
int main()
{
    int n;
    cin >> n;
    vector<std::vector<double>> A = init(n);
    ProcessMatrix(A, A.size(), "A");
    ProcessMatrix_OP(A, A.size(), "A");
    ProcessMatrix_SIMD(A, A.size(), "A");
    ProcessMatrix_SO(A, A.size(), "A");
    cout << endl;
    return 0;
}