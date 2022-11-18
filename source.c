#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifdef WITH_AVX2

#include <immintrin.h>
#include <omp.h>

#endif
#ifdef _OPENMP
#include <omp.h>
#endif


Matrix matmul_plain(const Matrix *matrix1, const Matrix *matrix2) {//矩阵相乘
    Matrix ans;
    if (matrix1->data == NULL || matrix2->data == NULL) {
        printf("The matrixs are not valid! \n");
        ans.data = NULL;
    } else if (matrix1->column != matrix2->row) {
        printf("The two matrixs doesn't match! \n");
        ans.data = NULL;
    } else {
        initialMatrix(&ans, matrix1->row, matrix2->column);
        float temp = 0;
        for (int i = 0; i < matrix1->row; ++i) {
            for (int j = 0; j < matrix2->column; ++j) {
                for (int k = 0; k < matrix1->column; ++k) {
                    temp += matrix1->data[i][k] * matrix2->data[k][j];
                }
                ans.data[i][j] = temp;
                temp = 0;
            }
        }
    }
    return ans;
}


Matrix matmul_improved(const Matrix *matrix1, const Matrix *matrix2) {
#ifdef WITH_AVX2
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();
    Matrix ans;

    initialMatrix(&ans, matrix1->row, matrix2->column);

    float *p1 = (float *) malloc(sizeof(float) * matrix1->column* matrix1->column);
    float *p2 = (float *) malloc(sizeof(float) * matrix2->column* matrix1->column);

        int o=0;
        int m=0;
        for (int i = 0; i < matrix1->row; ++i) {               
            for (int k = 0; k < matrix1->column; k+=1) {
                 p1[m++] = matrix1->data[i][k];
                // p1[m++] = matrix1->data[i][k+1];
                // p1[m++] = matrix1->data[i][k+2];
                // p1[m++] = matrix1->data[i][k+3];
                // p1[m++] = matrix1->data[i][k+4];
                // p1[m++] = matrix1->data[i][k+5];
                // p1[m++] = matrix1->data[i][k+6];
                // p1[m++] = matrix1->data[i][k+7];
            }
        }


//  #pragma omp parallel
        for (int j = 0; j < matrix2->column; ++j) {
            for (int k = 0; k < matrix1->column; k += 1) {
                p2[o++] = matrix1->data[k][j];
            //     p2[o++] = matrix1->data[k+1][j];
            //     p2[o++] = matrix1->data[k+2][j];
            //    p2[o++] = matrix1->data[k+3][j];
            //    p2[o++] = matrix1->data[k+4][j];
            //     p2[o++] = matrix1->data[k+5][j];
            //     p2[o++] = matrix1->data[k+6][j];
            //     p2[o++] = matrix1->data[k+7][j];
            }
        } 



#pragma omp parallel
    for (int i = 0; i < matrix1->row; ++i) {
        for (int j = 0; j < matrix2->column; ++j) {
            float temp = 0;
            float sum[8] = {0};
            __m256 c = _mm256_setzero_ps();
            ans.data[i][j] =0;

            int i1=i*matrix1->column;
            int j1=j*matrix1->column;
            for (int k = 0; k < matrix1->column / 8 * 8; k += 8) {
                //Use _mm256_mul_ps and _mm256_add_ps to process 8 elements at a time.


                // matrix1->data[i][k]
                //  matrix2->data[k][j]
                a = _mm256_loadu_ps(p1+i1+ k);
                b = _mm256_loadu_ps(p2+j1+ k);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));

            }
            _mm256_storeu_ps(sum, c);

            ans.data[i][j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

            for (int k = matrix1->column / 8 * 8; k < matrix1->column; k += 1) {
                //Tail case
                temp += matrix1->data[i][k] * matrix2->data[k][j];
            }
            ans.data[i][j]+=temp;
            temp=0;
        }
    }
    return ans;


#else
    printf( "AVX2 is not supported" );
#endif
}



void initialMatrix(Matrix *matrix, const int row, const int col) {//初始化一个元素都为0的矩阵
    if (row <= 0 || col <= 0) {
        printf("invalid matrix, please put in again! \n");

    } else {
        matrix->row = row;
        matrix->column = col;
        matrix->data = (float **) malloc(sizeof(float *) * row);//申请行空间
        if (matrix->data == NULL) {//申请空间失败
            printf("The memory is failed to allocated\n");
        }
        for (int i = 0; i < row; i++) {
            matrix->data[i] = (float *) malloc(sizeof(float) * col);//申请列空间
            if (matrix->data[i] == NULL) {
                printf("The memory is failed to allocated\n");
            }
            for (int j = 0; j < col; j++) {
                matrix->data[i][j] = 0;
            }
        }
    }
}

void createRamMatrix(Matrix *matrix, const int row, const int col, const int databound) {//检查合法性
    srand((unsigned) time(NULL));
    if (row <= 0) {
        printf("invalid matrix, please put in again! \n");
    } else {
        matrix->row = row;
        matrix->column = col;
        matrix->data = (float **) malloc(sizeof(float *) * row);//申请行空间
        if (matrix->data == NULL) {
            printf("The memory is failed to allocated\n");
        }
        for (int i = 0; i < row; i++) {
            matrix->data[i] = (float *) malloc(sizeof(float) * col);//申请列空间
            if (matrix->data[i] == NULL) {
                printf("The memory is failed to allocated\n");
            }

            for (int j = 0; j < col; j++) {
                float ram = rand() % databound + 1;
                matrix->data[i][j] = ram;
            }
        }
    }
    // printf("\n");
}
void showMatrix(const Matrix *matrix) {//打印矩阵
    if (matrix->data == NULL) {
        printf("The matrix is null!");
    } else {
        for (int i = 0; i < matrix->row; ++i) {
            for (int j = 0; j < matrix->column; ++j) {
                printf("%-6g", matrix->data[i][j]);
            }
            printf("\n");
        }
    }
    printf("\n");

}
