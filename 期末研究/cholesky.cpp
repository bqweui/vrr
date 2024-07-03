#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include<Windows.h>
#include <omp.h> // ����OpenMP֧��
#include <immintrin.h> // AVX intrinsics
using namespace std;

// ��ӡ����
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// ת�þ���
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

// ����˷�
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
#pragma omp parallel for collapse(2) // ���л�˫����ѭ��
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum; // ���½������
        }
    }

    return C;
}

vector<vector<double>> multiply_simd_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
#pragma omp parallel for collapse(2) // ���л�˫����ѭ��
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // ��ʼ��Ϊ0��256λ����
            int k = 0;
            for (; k <= n - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ�أ�ֱ��ʣ������4��Ԫ��
                // �Ӿ���A�м���4��������doubleֵ
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);
                // �Ӿ���B�й���һ����Ӧ��4��doubleֵ��������������Ҫ�ֶ�����
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]);
                // ��Ԫ����˲��ۼ�
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }
            // ����������AVX�����ۼӵ�C[i][j]
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            C[i][j] += temp[0] + temp[1] + temp[2] + temp[3];

            for (; k < n; ++k) { // ����ʣ��Ĳ���4������Ԫ��
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
            __m256d sum_vec = _mm256_setzero_pd(); // ��ʼ��Ϊ0��256λ����
            int k = 0;
            for (; k <= n - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ�أ�ֱ��ʣ������4��Ԫ��
                // �Ӿ���A�м���4��������doubleֵ
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]);
                // �Ӿ���B�й���һ����Ӧ��4��doubleֵ��������������Ҫ�ֶ�����
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]);
                // ��Ԫ����˲��ۼ�
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }
            // ����������AVX�����ۼӵ�C[i][j]
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            C[i][j] += temp[0] + temp[1] + temp[2] + temp[3];

            for (; k < n; ++k) { // ����ʣ��Ĳ���4������Ԫ��
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Cholesky�ֽ�
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

        // ���������
        if (L[i][i] <= 0) {
            throw runtime_error("������������");
        }
    }

    return L;
}

vector<vector<double>> cholesky_openmp(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        // ���л��Ŀ�������Ҫ��������ѭ��
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

        // ���������
        if (L[i][i] <= 0) {
            throw std::runtime_error("������������");
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
            __m256d sum_vec = _mm256_setzero_pd(); // ʹ��AVXָ���ʼ��Ϊ0��256λ����
            int k = 0;
            for (; k <= j - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ��
                __m256d l_i_vec = _mm256_loadu_pd(&L[i][k]);
                __m256d l_j_vec = _mm256_loadu_pd(&L[j][k]);
                __m256d prod_vec = _mm256_mul_pd(l_i_vec, l_j_vec); // ��Ԫ�����
#pragma omp atomic read
                sum_vec = _mm256_add_pd(sum_vec, prod_vec); // �ۼӵ�sum_vec��
            }
            // ����������AVX�����ۼӵ�sum
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            double sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < j; ++k) { // ����ʣ��Ĳ���4������Ԫ��
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
        // ���������
        if (L[i][i] <= 0) {
            throw std::runtime_error("������������");
        }
    }
    return L;
}
vector<vector<double>> cholesky_simd(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // ʹ��AVXָ���ʼ��Ϊ0��256λ����
            int k = 0;
            for (; k <= j - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ��
                __m256d l_i_vec = _mm256_loadu_pd(&L[i][k]);
                __m256d l_j_vec = _mm256_loadu_pd(&L[j][k]);
                __m256d prod_vec = _mm256_mul_pd(l_i_vec, l_j_vec); // ��Ԫ�����
                sum_vec = _mm256_add_pd(sum_vec, prod_vec); // �ۼӵ�sum_vec��
            }
            // ����������AVX�����ۼӵ�sum
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            double sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < j; ++k) { // ����ʣ��Ĳ���4������Ԫ��
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            }
            else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
        // ���������
        if (L[i][i] <= 0) {
            throw std::runtime_error("������������");
        }
    }
    return L;
}
void testCholesky(const vector<vector<double>>& A) {
    // ����Cholesky�ֽ�
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "ch" << " " << (tail - head) * 1000.0 / freq << endl;

    // ��ӡ���
   /* printMatrix(A);

    cout << "���� L :" << endl;
    printMatrix(L);

    cout << "���� L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // ��֤ L * L^T �Ƿ���� A
    vector<vector<double>> LLT = multiply(L, LT);
 /*   cout << "���� L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_simd(const vector<vector<double>>& A) {
    // ����Cholesky�ֽ�
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_simd(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "S" << " " << (tail - head) * 1000.0 / freq << endl;
    // ��ӡ���
    /*printMatrix(A);

    cout << "���� L :" << endl;
    printMatrix(L);

    cout << "���� L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // ��֤ L * L^T �Ƿ���� A
    vector<vector<double>> LLT = multiply_simd(L, LT);
    /*cout << "���� L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_openmp(const vector<vector<double>>& A) {
    // ����Cholesky�ֽ�
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_openmp(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "op" << " " << (tail - head) * 1000.0 / freq << endl;

    // ��ӡ���
    /*printMatrix(A);

    cout << "���� L :" << endl;
    printMatrix(L);

    cout << "���� L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // ��֤ L * L^T �Ƿ���� A
    vector<vector<double>> LLT = multiply_openmp(L, LT);
    /*cout << "���� L * L^T:" << endl;
    printMatrix(LLT);*/

}
void testCholesky_simd_openmp(const vector<vector<double>>& A) {
    // ����Cholesky�ֽ�
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    vector<vector<double>> L = cholesky_simd_openmp(A);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SO" << " " << (tail - head) * 1000.0 / freq << endl;

    // ��ӡ���
    /*printMatrix(A);

    cout << "���� L :" << endl;
    printMatrix(L);

    cout << "���� L^T:" << endl;*/
    vector<vector<double>> LT = transpose(L);
    /*printMatrix(LT);*/

    // ��֤ L * L^T �Ƿ���� A
    vector<vector<double>> LLT = multiply_simd_openmp(L, LT);
    /*cout << "���� L * L^T:" << endl;
    printMatrix(LLT);*/

}
vector<vector<double>> init(int n) {
    srand(time(0));
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand() % 100 + 1; // ���� 1 �� 100 ��Χ�ڵ������
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