#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include<Windows.h>
#include <omp.h> // 引入OpenMP支持
#include <immintrin.h> // AVX intrinsics
using namespace std;

// 打印矩阵
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// 转置矩阵
vector<vector<double>> transpose(const vector<vector<double>>& A) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double>> AT(n, vector<double>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

// 矩阵乘法
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<double>> multiply_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
#pragma omp parallel for collapse(2) // 并行化双层外循环
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum; // 更新结果矩阵
        }
    }

    return C;
}

vector<vector<double>> multiply_simd_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
#pragma omp parallel for collapse(2) // 并行化双层外循环
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 初始化为0的256位向量
            int k = 0;
            for (; k <= n - 4; k += 4) { // 使用AVX一次处理4个元素，直到剩下少于4个元素
                // 从矩阵A中加载4个连续的double值
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);
                // 从矩阵B中构造一个对应的4个double值的向量，这里需要手动设置
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]);
                // 逐元素相乘并累加
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }
            // 将计算结果从AVX向量累加到C[i][j]
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            C[i][j] += temp[0] + temp[1] + temp[2] + temp[3];

            for (; k < n; ++k) { // 处理剩余的不是4倍数的元素
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
vector<vector<double>> multiply_simd(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 初始化为0的256位向量
            int k = 0;
            for (; k <= n - 4; k += 4) { // 使用AVX一次处理4个元素，直到剩下少于4个元素
                // 从矩阵A中加载4个连续的double值
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);
                // 从矩阵B中构造一个对应的4个double值的向量，这里需要手动设置
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]);
                // 逐元素相乘并累加
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }
            // 将计算结果从AVX向量累加到C[i][j]
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            C[i][j] += temp[0] + temp[1] + temp[2] + temp[3];

            for (; k < n; ++k) { // 处理剩余的不是4倍数的元素
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Cholesky分解
vector<vector<double>> cholesky(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            }
            else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }

        // 检查正定性
        if (L[i][i] <= 0) {
            throw runtime_error("矩阵不是正定的");
        }
    }

    return L;
}

vector<vector<double>> cholesky_openmp(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        // 并行化的可能性主要在这个外层循环
#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j <= i; ++j) {
            double sum = 0;
            for (int k = 0; k < j; ++k) {
#pragma omp atomic read
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            }
            else {
#pragma omp atomic read
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }

        // 检查正定性
        if (L[i][i] <= 0) {
            throw std::runtime_error("矩阵不是正定的");
        }
    }

    return L;
}
vector<vector<double>> cholesky_simd_openmp(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j <= i; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 使用AVX指令初始化为0的256位向量
            int k = 0;
            for (; k <= j - 4; k += 4) { // 使用AVX一次处理4个元素
                __m256d l_i_vec = _mm256_loadu_pd(&L[i][k]);
                __m256d l_j_vec = _mm256_loadu_pd(&L[j][k]);
                __m256d prod_vec = _mm256_mul_pd(l_i_vec, l_j_vec); // 逐元素相乘
#pragma omp atomic read
                sum_vec = _mm256_add_pd(sum_vec, prod_vec); // 累加到sum_vec中
            }
            // 将计算结果从AVX向量累加到sum
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            double sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < j; ++k) { // 处理剩余的不是4倍数的元素
#pragma omp atomic read
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            }
            else {
#pragma omp atomic read
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
        // 检查正定性
        if (L[i][i] <= 0) {
            throw std::runtime_error("矩阵不是正定的");
        }
    }
    return L;
}
vector<vector<double>> cholesky_simd(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 使用AVX指令初始化为0的256位向量
            int k = 0;
            for (; k <= j - 4; k += 4) { // 使用AVX一次处理4个元素
                __m256d l_i_vec = _mm256_loadu_pd(&L[i][k]);
                __m256d l_j_vec = _mm256_loadu_pd(&L[j][k]);
                __m256d prod_vec = _mm256_mul_pd(l_i_vec, l_j_vec); // 逐元素相乘
                sum_vec = _mm256_add_pd(sum_vec, prod_vec); // 累加到sum_vec中
            }
            // 将计算结果从AVX向量累加到sum
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            double sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < j; ++k) { // 处理剩余的不是4倍数的元素
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            }
            else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
        // 检查正定性
        if (L[i][i] <= 0) {
            throw std::runtime_error("矩阵不是正定的");
        }
    }
    return L;
}
void testCholesky(const vector<vector<double>>& A) {
    // 进行Cholesky分解
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "ch" << " " << (tail - head) * 1000.0 / freq << endl;

    // 打印结果
   /* printMatrix(A);

    cout << "矩阵 L :" << endl;
    printMatrix(L);

    cout << "矩阵 L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // 验证 L * L^T 是否等于 A
    vector<vector<double>> LLT = multiply(L, LT);
 /*   cout << "矩阵 L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_simd(const vector<vector<double>>& A) {
    // 进行Cholesky分解
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_simd(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "S" << " " << (tail - head) * 1000.0 / freq << endl;
    // 打印结果
    /*printMatrix(A);

    cout << "矩阵 L :" << endl;
    printMatrix(L);

    cout << "矩阵 L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // 验证 L * L^T 是否等于 A
    vector<vector<double>> LLT = multiply_simd(L, LT);
    /*cout << "矩阵 L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_openmp(const vector<vector<double>>& A) {
    // 进行Cholesky分解
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_openmp(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "op" << " " << (tail - head) * 1000.0 / freq << endl;

    // 打印结果
    /*printMatrix(A);

    cout << "矩阵 L :" << endl;
    printMatrix(L);

    cout << "矩阵 L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // 验证 L * L^T 是否等于 A
    vector<vector<double>> LLT = multiply_openmp(L, LT);
    /*cout << "矩阵 L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_simd_openmp(const vector<vector<double>>& A) {
    // 进行Cholesky分解
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_simd_openmp(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SO" << " " << (tail - head) * 1000.0 / freq << endl;

    // 打印结果
    /*printMatrix(A);

    cout << "矩阵 L :" << endl;
    printMatrix(L);

    cout << "矩阵 L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // 验证 L * L^T 是否等于 A
    vector<vector<double>> LLT = multiply_simd_openmp(L, LT);
    /*cout << "矩阵 L * L^T:" << endl;
    printMatrix(LLT);*/

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
int main() {
    int n;
    cin >> n;
    vector<std::vector<double>> A = init(n);
    testCholesky(A);
    testCholesky_openmp(A);
    testCholesky_simd(A);
    testCholesky_simd_openmp(A);
    cout << endl;

    return 0;
}