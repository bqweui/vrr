#include <iostream>
#include <vector>
#include <cmath>
#include<Windows.h>
#include <immintrin.h> // ����AVX֧��
#include <omp.h> // ����OpenMP֧��
using namespace std;
// ����������2����
double norm(const vector<double>& v) {
    double sum = 0;
    for (double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}
// ��������
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
#pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
vector<double> vector_sub_simd_openmp(const vector<double>& a, const vector<double>& b) {
    size_t size = a.size();
    vector<double> result(size);
    int i;
#pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // һ�μ�����������a��b�е�4��Ԫ��
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        // ִ����Ԫ���������
        __m256d vresult = _mm256_sub_pd(va, vb);
        // ������洢��result����
        _mm256_storeu_pd(&result[i], vresult);
    }
    // ����ʣ��Ԫ��
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
        // һ�μ�����������a��b�е�4��Ԫ��
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        // ִ����Ԫ���������
        __m256d vresult = _mm256_sub_pd(va, vb);
        // ������洢��result����
        _mm256_storeu_pd(&result[i], vresult);
    }
    // ����ʣ��Ԫ��
    for (; i < static_cast<int>(size); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}
// �������
double dot(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_openmp(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    #pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_simd_openmp(vector<double>& a, const vector<double>& b) {
    int size = a.size();
    __m256d sum_vec = _mm256_setzero_pd(); // ʹ��AVXָ���ʼ��Ϊ0��256λ����
    int i;
#pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (i = 0; i <= size - 4; i += 4) { // ����AVXһ�δ���4��Ԫ��
        __m256d va = _mm256_loadu_pd(&a[i]); // ������a����4��˫���ȸ�����
        __m256d vb = _mm256_loadu_pd(&b[i]); // ������b����4��˫���ȸ�����
        __m256d prod = _mm256_mul_pd(va, vb); // ��Ԫ�����
        sum_vec = _mm256_add_pd(sum_vec, prod); // �ۼӵ�sum_vec��
    }
    double result_array[4];
    _mm256_storeu_pd(result_array, sum_vec); // ������sum_vec�����ݴ洢����ͨ������
    double result = result_array[0] + result_array[1] + result_array[2] + result_array[3]; // ���ۼӽ����͵õ��������
    // ����ʣ�಻��4����Ԫ��
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
double dot_simd(vector<double>& a, const vector<double>& b) {
    int size = a.size();
    __m256d sum_vec = _mm256_setzero_pd(); // ʹ��AVXָ���ʼ��Ϊ0��256λ����
    int i;
    for (i = 0; i <= size - 4; i += 4) { // ����AVXһ�δ���4��Ԫ��
        __m256d va = _mm256_loadu_pd(&a[i]); // ������a����4��˫���ȸ�����
        __m256d vb = _mm256_loadu_pd(&b[i]); // ������b����4��˫���ȸ�����
        __m256d prod = _mm256_mul_pd(va, vb); // ��Ԫ�����
        sum_vec = _mm256_add_pd(sum_vec, prod); // �ۼӵ�sum_vec��
    }
    double result_array[4];
    _mm256_storeu_pd(result_array, sum_vec); // ������sum_vec�����ݴ洢����ͨ������
    double result = result_array[0] + result_array[1] + result_array[2] + result_array[3]; // ���ۼӽ����͵õ��������
    // ����ʣ�಻��4����Ԫ��
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
// �������Ա���
vector<double> scalar_multiply(const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_openmp (const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    #pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_simd_openmp(const vector<double>& v, double scalar) {
    size_t size = v.size();
    vector<double> result(size);
    // ʹ��_broadcast_sdָ����ر���ֵ��һ��AVX�����У�������������Ԫ�ض����������ֵ
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    int i;
#pragma omp parallel for // ָʾOpenMP���л����forѭ��
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // ������v����4��˫���ȸ�����
        __m256d vec = _mm256_loadu_pd(&v[i]);
        // ������������
        __m256d prod = _mm256_mul_pd(vec, scalar_vec);
        // ������洢��result����
        _mm256_storeu_pd(&result[i], prod);
    }
    // ����ʣ��Ԫ��
    for (; i < static_cast<int>(size); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
vector<double> scalar_multiply_simd(const vector<double>& v, double scalar) {
    size_t size = v.size();
    vector<double> result(size);
    // ʹ��_broadcast_sdָ����ر���ֵ��һ��AVX�����У�������������Ԫ�ض����������ֵ
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    int i;
    for (i = 0; i <= static_cast<int>(size) - 4; i += 4) {
        // ������v����4��˫���ȸ�����
        __m256d vec = _mm256_loadu_pd(&v[i]);
        // ������������
        __m256d prod = _mm256_mul_pd(vec, scalar_vec);
        // ������洢��result����
        _mm256_storeu_pd(&result[i], prod);
    }
    // ����ʣ��Ԫ��
    for (; i < static_cast<int>(size); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}
// Gram-Schmidt����������
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
// ����ת��
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
// ����˷�
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
#pragma omp parallel for collapse(2) // ���л�˫����ѭ��
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum; // ���½������ע��������ݾ���
        }
    }

    return result;
}
vector<std::vector<double>> matrix_multiply_simd_openmp(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> result(n, vector<double>(m, 0));
#pragma omp parallel for collapse(2) // ���л�˫����ѭ��
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            __m256d sum_vec = _mm256_setzero_pd(); // ��ʼ��Ϊ0��256λ����
            for (int k = 0; k <= p - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ��
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]); // ��A�м���4��������doubleֵ
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]); // ��B�м���4����Ӧ��doubleֵ
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec)); // ��Ԫ����˲��ۼ�
            }
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec); // ������sum_vec�����ݴ洢����ʱ������
            result[i][j] = temp[0] + temp[1] + temp[2] + temp[3]; // ���ۼӽ����͵õ��������

            // ����ʣ��Ԫ��
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
            __m256d sum_vec = _mm256_setzero_pd(); // ��ʼ��Ϊ0��256λ����
            for (int k = 0; k <= p - 4; k += 4) { // ʹ��AVXһ�δ���4��Ԫ��
                __m256d a_vec = _mm256_loadu_pd(&A[i][k]); // ��A�м���4��������doubleֵ
                __m256d b_vec = _mm256_set_pd(B[k + 3][j], B[k + 2][j], B[k + 1][j], B[k][j]); // ��B�м���4����Ӧ��doubleֵ
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec)); // ��Ԫ����˲��ۼ�
            }
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec); // ������sum_vec�����ݴ洢����ʱ������
            result[i][j] = temp[0] + temp[1] + temp[2] + temp[3]; // ���ۼӽ����͵õ��������

            // ����ʣ��Ԫ��
            for (int k = p - (p % 4); k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
// ��С����ֵ����ֵ����Ϊ��
void round_to_zero(vector<vector<double>>& matrix, double threshold = 1e-10) {
    for (auto& row : matrix) {
        for (auto& val : row) {
            if (abs(val) < threshold) {
                val = 0.0;
            }
        }
    }
}
// ��ӡ����
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
    // ת��A�Ա�������������
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "PM" << " " << (tail - head) * 1000.0 / freq << endl;
    // ת��Q�Իָ�ԭʼ�������״
    Q = transpose(Q);
    /*cout << "���� " << name << ":" << endl;
    print_matrix(A);
    cout << "���� Q:" << endl;
    print_matrix(Q);
    cout << "���� R:" << endl;
    print_matrix(R);*/
    // ��֤ Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply(Q_T, Q);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(Q_TQ);
    /*cout << "���� Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // ��֤ Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply(Q, Q_T);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QQ_T);
    /*cout << "���� Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // ��֤ Q * R = A
    vector<vector<double>> QR = matrix_multiply(Q, R);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QR);
   /* cout << "���� Q * R:" << endl;
    print_matrix(QR);
    cout << "ԭʼ���� :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_SIMD(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // ת��A�Ա�������������
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_SIMD(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "S" << " " << (tail - head) * 1000.0 / freq << endl;
    // ת��Q�Իָ�ԭʼ�������״
    Q = transpose(Q);
    /*cout << "���� " << name << ":" << endl;
    print_matrix(A);
    cout << "���� Q:" << endl;
    print_matrix(Q);
    cout << "���� R:" << endl;
    print_matrix(R);*/
    // ��֤ Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_simd(Q_T, Q);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(Q_TQ);
    /*cout << "���� Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // ��֤ Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_simd(Q, Q_T);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QQ_T);
    /*cout << "���� Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // ��֤ Q * R = A
    vector<vector<double>> QR = matrix_multiply_simd(Q, R);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QR);
    /*cout << "���� Q * R:" << endl;
    print_matrix(QR);
    cout << "ԭʼ���� :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_openmp(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // ת��A�Ա�������������
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_openmp(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "OP" << " " << (tail - head) * 1000.0 / freq << endl;
    // ת��Q�Իָ�ԭʼ�������״
    Q = transpose(Q);
    /*cout << "���� " << name << ":" << endl;
    print_matrix(A);
    cout << "���� Q:" << endl;
    print_matrix(Q);
    cout << "���� R:" << endl;
    print_matrix(R);*/
    // ��֤ Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_openmp(Q_T, Q);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(Q_TQ);
    /*cout << "���� Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // ��֤ Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_openmp(Q, Q_T);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QQ_T);
    /*cout << "���� Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // ��֤ Q * R = A
    vector<vector<double>> QR = matrix_multiply_openmp(Q, R);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QR);
    /*cout << "���� Q * R:" << endl;
    print_matrix(QR);
    cout << "ԭʼ���� :" << endl;
    print_matrix(A);*/
}
void ProcessMatrix_SIMD_openmp(const vector<vector<double>>& A, const string& name) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> Q(n, vector<double>(m, 0));
    vector<vector<double>> R(m, vector<double>(m, 0));
    // ת��A�Ա�������������
    vector<vector<double>> A_T = transpose(A);
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    gram_schmidt_SIMD_openmp(A_T, Q, R);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SO" << " " << (tail - head) * 1000.0 / freq << endl;
    // ת��Q�Իָ�ԭʼ�������״
    Q = transpose(Q);
    /*cout << "���� " << name << ":" << endl;
    print_matrix(A);
    cout << "���� Q:" << endl;
    print_matrix(Q);
    cout << "���� R:" << endl;
    print_matrix(R);*/
    // ��֤ Q^T Q = I
    vector<vector<double>> Q_T = transpose(Q);
    vector<vector<double>> Q_TQ = matrix_multiply_simd_openmp(Q_T, Q);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(Q_TQ);
    /*cout << "���� Q^T Q:" << endl;
    print_matrix(Q_TQ);*/
    // ��֤ Q Q^T = I
    vector<vector<double>> QQ_T = matrix_multiply_simd_openmp(Q, Q_T);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QQ_T);
    /*cout << "���� Q Q^T:" << endl;
    print_matrix(QQ_T);*/
    // ��֤ Q * R = A
    vector<vector<double>> QR = matrix_multiply_simd_openmp(Q, R);
    // ��С����ֵ����ֵ����Ϊ��
    round_to_zero(QR);
    /*cout << "���� Q * R:" << endl;
    print_matrix(QR);
    cout << "ԭʼ���� :" << endl;
    print_matrix(A);*/
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
    ProcessMatrix(A, "A");
    ProcessMatrix_openmp(A, "A");
    ProcessMatrix_SIMD(A, "A");
    ProcessMatrix_SIMD_openmp(A, "A");
    return 0;
}