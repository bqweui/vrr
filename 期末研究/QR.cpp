#include <iostream>
#include <vector>
#include <cmath>
#include<Windows.h>
#include <immintrin.h> // 包含AVX支持
#include <omp.h> // 引入OpenMP支持
using namespace std;
// 计算向量的2范数
double norm(const vector<double>& v) {
    double sum = 0;
    for (double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}
// 向量减法
vector<double> vector_sub(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
vector<double> vector_sub_openmp(const vector<double>& a, const vector<double>& b) {
    size_t size = a.size();
    vector<double> result(size);
#pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
vector<double> vector_sub_simd_openmp(const vector<double>& a, const vector<double>& b) {
    size_t size = a.size();
    vector<double> result(size);
    int i;
#pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // 一次加载两个向量a和b中的4个元素
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        // 执行逐元素相减操作
        __m256d vresult = _mm256_sub_pd(va, vb);
        // 将结果存储回result向量
        _mm256_storeu_pd(&result[i], vresult);
    }
    // 处理剩余元素
    for (; i < static_cast<int>(size); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
vector<double> vector_sub_simd(const vector<double>& a, const vector<double>& b) {
    size_t size = a.size();
    vector<double> result(size);
    int i;
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // 一次加载两个向量a和b中的4个元素
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        // 执行逐元素相减操作
        __m256d vresult = _mm256_sub_pd(va, vb);
        // 将结果存储回result向量
        _mm256_storeu_pd(&result[i], vresult);
    }
    // 处理剩余元素
    for (; i < static_cast<int>(size); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
// 向量点乘
double dot(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_openmp(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    #pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_simd_openmp(vector<double>& a, const vector<double>& b) {
    int size = a.size();
    __m256d sum_vec = _mm256_setzero_pd(); // 使用AVX指令初始化为0的256位向量
    int i;
#pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (i = 0; i <= size - 4; i += 4) { // 利用AVX一次处理4个元素
        __m256d va = _mm256_loadu_pd(&a[i]); // 从向量a加载4个双精度浮点数
        __m256d vb = _mm256_loadu_pd(&b[i]); // 从向量b加载4个双精度浮点数
        __m256d prod = _mm256_mul_pd(va, vb); // 逐元素相乘
        sum_vec = _mm256_add_pd(sum_vec, prod); // 累加到sum_vec中
    }
    double result_array[4];
    _mm256_storeu_pd(result_array, sum_vec); // 将向量sum_vec的内容存储到普通数组中
    double result = result_array[0] + result_array[1] + result_array[2] + result_array[3]; // 将累加结果求和得到标量结果
    // 处理剩余不足4个的元素
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_simd(vector<double>& a, const vector<double>& b) {
    int size = a.size();
    __m256d sum_vec = _mm256_setzero_pd(); // 使用AVX指令初始化为0的256位向量
    int i;
    for (i = 0; i <= size - 4; i += 4) { // 利用AVX一次处理4个元素
        __m256d va = _mm256_loadu_pd(&a[i]); // 从向量a加载4个双精度浮点数
        __m256d vb = _mm256_loadu_pd(&b[i]); // 从向量b加载4个双精度浮点数
        __m256d prod = _mm256_mul_pd(va, vb); // 逐元素相乘
        sum_vec = _mm256_add_pd(sum_vec, prod); // 累加到sum_vec中
    }
    double result_array[4];
    _mm256_storeu_pd(result_array, sum_vec); // 将向量sum_vec的内容存储到普通数组中
    double result = result_array[0] + result_array[1] + result_array[2] + result_array[3]; // 将累加结果求和得到标量结果
    // 处理剩余不足4个的元素
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
// 向量乘以标量
vector<double> scalar_multiply(const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_openmp (const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    #pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_simd_openmp(const vector<double>& v, double scalar) {
    size_t size = v.size();
    vector<double> result(size);
    // 使用_broadcast_sd指令加载标量值到一个AVX向量中，该向量的所有元素都是这个标量值
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    int i;
#pragma omp parallel for // 指示OpenMP并行化这个for循环
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // 从向量v加载4个双精度浮点数
        __m256d vec = _mm256_loadu_pd(&v[i]);
        // 与标量向量相乘
        __m256d prod = _mm256_mul_pd(vec, scalar_vec);
        // 将结果存储回result向量
        _mm256_storeu_pd(&result[i], prod);
    }
    // 处理剩余元素
    for (; i < static_cast<int>(size); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_simd(const vector<double>& v, double scalar) {
    size_t size = v.size();
    vector<double> result(size);
    // 使用_broadcast_sd指令加载标量值到一个AVX向量中，该向量的所有元素都是这个标量值
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    int i;
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // 从向量v加载4个双精度浮点数
        __m256d vec = _mm256_loadu_pd(&v[i]);
        // 与标量向量相乘
        __m256d prod = _mm256_mul_pd(vec, scalar_vec);
        // 将结果存储回result向量
        _mm256_storeu_pd(&result[i], prod);
    }
    // 处理剩余元素
    for (; i < static_cast<int>(size); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
// Gram-Schmidt正交化过程
void gram_schmidt(const vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = A.size();
    int m = A[0].size();
    for (int k = 0; k < m; ++k) {
        vector<double> u = A[k];
        for (int i = 0; i < k; ++i) {
            R[i][k] = dot(Q[i], A[k]);
            vector<double> proj = scalar_multiply(Q[i], R[i][k]);
            u = vector_sub(u, proj);
        }
        R[k][k] = norm(u);
        Q[k] = scalar_multiply(u, 1.0 / R[k][k]);
    }
}
void gram_schmidt_SIMD(const vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = A.size();
    int m = A[0].size();
    for (int k = 0; k < m; ++k) {
        vector<double> u = A[k];
        for (int i = 0; i < k; ++i) {
            R[i][k] = dot_simd(Q[i], A[k]);
            vector<double> proj = scalar_multiply_simd(Q[i], R[i][k]);
            u = vector_sub_simd(u, proj);
        }
        R[k][k] = norm(u);
        Q[k] = scalar_multiply_simd(u, 1.0 / R[k][k]);
    }
}
void gram_schmidt_openmp(const vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = A.size();
    int m = A[0].size();
    for (int k = 0; k < m; ++k) {
        vector<double> u = A[k];
        for (int i = 0; i < k; ++i) {
            R[i][k] = dot_openmp (Q[i], A[k]);
            vector<double> proj = scalar_multiply_openmp(Q[i], R[i][k]);
            u = vector_sub_openmp(u, proj);
        }
        R[k][k] = norm(u);
        Q[k] = scalar_multiply_openmp(u, 1.0 / R[k][k]);
    }
}
void gram_schmidt_SIMD_openmp(const vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = A.size();
    int m = A[0].size();
    for (int k = 0; k < m; ++k) {
        vector<double> u = A[k];
        for (int i = 0; i < k; ++i) {
            R[i][k] = dot_simd_openmp(Q[i], A[k]);
            vector<double> proj = scalar_multiply_simd_openmp(Q[i], R[i][k]);
            u = vector_sub_simd_openmp(u, proj);
        }
        R[k][k] = norm(u);
        Q[k] = scalar_multiply_simd_openmp(u, 1.0 / R[k][k]);
    }
}
// 矩阵转置
vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    int m = matrix[0].size();
    vector<vector<double>> result(m, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
// 矩阵乘法
vector<vector<double>> matrix_multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> result(n, vector<double>(m, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
vector<vector<double>> matrix_multiply_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> result(n, vector<double>(m, 0));
#pragma omp parallel for collapse(2) // 并行化双层外循环
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum; // 更新结果矩阵，注意避免数据竞争
        }
    }

    return result;
}
vector<std::vector<double>> matrix_multiply_simd_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> result(n, vector<double>(m, 0));
#pragma omp parallel for collapse(2) // 并行化双层外循环
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 初始化为0的256位向量
            for (int k = 0; k <= p - 4; k += 4) { // 使用AVX一次处理4个元素
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]); // 从A中加载4个连续的double值
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]); // 从B中加载4个对应的double值
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec)); // 逐元素相乘并累加
            }
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec); // 将向量sum_vec的内容存储到临时数组中
            result[i][j] = temp[0] + temp[1] + temp[2] + temp[3]; // 将累加结果求和得到标量结果

            // 处理剩余元素
            for (int k = p - (p % 4); k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
vector<std::vector<double>> matrix_multiply_simd(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> result(n, vector<double>(m, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // 初始化为0的256位向量
            for (int k = 0; k <= p - 4; k += 4) { // 使用AVX一次处理4个元素
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]); // 从A中加载4个连续的double值
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]); // 从B中加载4个对应的double值
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec)); // 逐元素相乘并累加
            }
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec); // 将向量sum_vec的内容存储到临时数组中
            result[i][j] = temp[0] + temp[1] + temp[2] + temp[3]; // 将累加结果求和得到标量结果

            // 处理剩余元素
            for (int k = p - (p % 4); k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
// 将小于阈值的数值设置为零
void round_to_zero(vector<vector<double>>& matrix, double threshold = 1e-10) {
    for (auto& row : matrix) {
        for (auto& val : row) {
            if (abs(val) < threshold) {
                val = 0.0;
            }
        }
    }
}
// 打印矩阵
void print_matrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}
void ProcessMatrix(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // 转置A以便于列向量操作
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "PM" << " " << (tail - head) * 1000.0 / freq << endl;
    // 转置Q以恢复原始矩阵的形状
    Q = transpose(Q);
    /*cout << "矩阵 " << name << ":" << endl;
    print_matrix(A);
    cout << "矩阵 Q:" << endl;
    print_matrix(Q);
    cout << "矩阵 R:" << endl;
    print_matrix(R);*/
    // 验证 Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply(Q_T, Q);
    // 将小于阈值的数值设置为零
    round_to_zero(Q_TQ);
    /*cout << "矩阵 Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // 验证 Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply(Q, Q_T);
    // 将小于阈值的数值设置为零
    round_to_zero(QQ_T);
    /*cout << "矩阵 Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // 验证 Q * R = A
    vector<vector<double>> QR = matrix_multiply(Q, R);
    // 将小于阈值的数值设置为零
    round_to_zero(QR);
   /* cout << "矩阵 Q * R:" << endl;
    print_matrix(QR);
    cout << "原始矩阵 :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_SIMD(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // 转置A以便于列向量操作
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_SIMD(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "S" << " " << (tail - head) * 1000.0 / freq << endl;
    // 转置Q以恢复原始矩阵的形状
    Q = transpose(Q);
    /*cout << "矩阵 " << name << ":" << endl;
    print_matrix(A);
    cout << "矩阵 Q:" << endl;
    print_matrix(Q);
    cout << "矩阵 R:" << endl;
    print_matrix(R);*/
    // 验证 Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_simd(Q_T, Q);
    // 将小于阈值的数值设置为零
    round_to_zero(Q_TQ);
    /*cout << "矩阵 Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // 验证 Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_simd(Q, Q_T);
    // 将小于阈值的数值设置为零
    round_to_zero(QQ_T);
    /*cout << "矩阵 Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // 验证 Q * R = A
    vector<vector<double>> QR = matrix_multiply_simd(Q, R);
    // 将小于阈值的数值设置为零
    round_to_zero(QR);
    /*cout << "矩阵 Q * R:" << endl;
    print_matrix(QR);
    cout << "原始矩阵 :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_openmp(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // 转置A以便于列向量操作
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_openmp(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "OP" << " " << (tail - head) * 1000.0 / freq << endl;
    // 转置Q以恢复原始矩阵的形状
    Q = transpose(Q);
    /*cout << "矩阵 " << name << ":" << endl;
    print_matrix(A);
    cout << "矩阵 Q:" << endl;
    print_matrix(Q);
    cout << "矩阵 R:" << endl;
    print_matrix(R);*/
    // 验证 Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_openmp(Q_T, Q);
    // 将小于阈值的数值设置为零
    round_to_zero(Q_TQ);
    /*cout << "矩阵 Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // 验证 Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_openmp(Q, Q_T);
    // 将小于阈值的数值设置为零
    round_to_zero(QQ_T);
    /*cout << "矩阵 Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // 验证 Q * R = A
    vector<vector<double>> QR = matrix_multiply_openmp(Q, R);
    // 将小于阈值的数值设置为零
    round_to_zero(QR);
    /*cout << "矩阵 Q * R:" << endl;
    print_matrix(QR);
    cout << "原始矩阵 :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_SIMD_openmp(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // 转置A以便于列向量操作
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_SIMD_openmp(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SO" << " " << (tail - head) * 1000.0 / freq << endl;
    // 转置Q以恢复原始矩阵的形状
    Q = transpose(Q);
    /*cout << "矩阵 " << name << ":" << endl;
    print_matrix(A);
    cout << "矩阵 Q:" << endl;
    print_matrix(Q);
    cout << "矩阵 R:" << endl;
    print_matrix(R);*/
    // 验证 Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_simd_openmp(Q_T, Q);
    // 将小于阈值的数值设置为零
    round_to_zero(Q_TQ);
    /*cout << "矩阵 Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // 验证 Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_simd_openmp(Q, Q_T);
    // 将小于阈值的数值设置为零
    round_to_zero(QQ_T);
    /*cout << "矩阵 Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // 验证 Q * R = A
    vector<vector<double>> QR = matrix_multiply_simd_openmp(Q, R);
    // 将小于阈值的数值设置为零
    round_to_zero(QR);
    /*cout << "矩阵 Q * R:" << endl;
    print_matrix(QR);
    cout << "原始矩阵 :" << endl;
    print_matrix(A);*/
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
    ProcessMatrix(A, "A");
    ProcessMatrix_openmp(A, "A");
    ProcessMatrix_SIMD(A, "A");
    ProcessMatrix_SIMD_openmp(A, "A");
    return 0;
}