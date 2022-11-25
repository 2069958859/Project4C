#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <immintrin.h>

#ifdef WITH_AVX2

#include <immintrin.h>
#include <omp.h>

#endif
#ifdef _OPENMP
#include <omp.h>
#endif


bool matmul_plain(const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans) {//矩阵相乘,一维

    if(matrix1==NULL||matrix2==NULL||ans==NULL){
        fprintf(stderr,"The matrixs are not valid! \n");//输出到错误流
        return false;
    }
    else if (matrix1->data == NULL || matrix2->data == NULL||ans->data==NULL) {
        fprintf(stderr,"The matrix data is not valid! \n");
        return false;
    } else if (matrix1->column != matrix2->row||matrix1->row != ans->row||matrix2->column!=ans->column) {
        fprintf(stderr,"The matrixs ans the output doesn't match! \n");
        return false;
    } else {
        float temp = 0;
        for (size_t i = 0; i < matrix1->row; ++i) {
            for (size_t j = 0; j < matrix2->column; ++j) {
                for (size_t k = 0; k < matrix1->column; ++k) {
                    temp += matrix1->data[i * matrix1->row + k] * matrix2->data[k * matrix1->column + j];
                }
                ans->data[i * matrix1->row + j] = temp;
                temp = 0;
            }
        }
        return true;
    }
}


        

// bool matmul_improved(const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans) {//矩阵相乘,一维
// #ifdef WITH_AVX2
//     __m256 a, b;
//     __m256 c = _mm256_setzero_ps();

//     if(matrix1==NULL||matrix2==NULL||ans==NULL){
//         fprintf(stderr,"The matrixs are not valid! \n");//输出到错误流
//         return false;
//     }
//     else if (matrix1->data == NULL || matrix2->data == NULL||ans->data==NULL) {
//         fprintf(stderr,"The matrix data is not valid! \n");
//         return false;
//     } else if (matrix1->column != matrix2->row||matrix1->row != ans->row||matrix2->column!=ans->column) {
//         return false;
//     } else {
//         float *p2 = (float *) malloc(sizeof(float) * matrix2->column * matrix1->column);
//         if (p2 == NULL) {//申请空间失败
//             fprintf(stderr,"p2 is failed to allocated\n");
//             return false;
//         }

//         size_t o = 0;


//         for (size_t j = 0; j < matrix2->column; ++j) {
//             for (size_t k = 0; k < matrix1->column / 8 * 8; k += 8) {
//                 p2[o++] = matrix2->data[k * matrix2->column + j];
//                 p2[o++] = matrix2->data[(1 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(2 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(3 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(4 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(5 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(6 + k) * matrix2->column + j];
//                 p2[o++] = matrix2->data[(7 + k) * matrix2->column + j];
//             }
//             for (size_t k = matrix1->column / 8 * 8; k < matrix1->column; k++) {
//                 p2[o++] = matrix2->data[k * matrix2->column + j];

//             }
//         }


// #pragma omp parallel for
//         for (size_t i = 0; i < matrix1->row; ++i) {
//             for (size_t j = 0; j < matrix2->column; ++j) {
//                 float temp = 0;
//                 float sum[8] = {0};
//                 __m256 c = _mm256_setzero_ps();
//                 ans->data[i * matrix1->row + j] = 0;

//                 size_t i1 = i * matrix1->row;
//                 size_t j1 = j * matrix2->column;

//                 for (size_t k = 0; k < matrix1->column / 8 * 8; k += 8) {
//                     a = _mm256_loadu_ps(&matrix1->data[0] + i1 + k);
//                     b = _mm256_loadu_ps(p2 + j1 + k);
//                     c = _mm256_add_ps(c, _mm256_mul_ps(a, b));

//                 }
//                 _mm256_storeu_ps(sum, c);

//                 ans->data[i * matrix1->row + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

//                 for (size_t k = matrix1->column / 8 * 8; k < matrix1->column; k += 1) {
//                     //Tail case
//                     temp += matrix1->data[i1 + k] * matrix2->data[k * matrix1->column + j];
//                 }
//                 ans->data[i1 + j] += temp;
//                 temp = 0;
//             }
//             // printf("%d\n",i);
//         }
//         free(p2);//释放空间
//         return true;
//     }

// #else
//     printf("AVX2 is not supported");
// #endif

// }


bool matmul_improved(const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans) {//矩阵相乘,一维
// #ifdef WITH_AVX2
    __m256 a, b;
    // __m256 c = _mm256_setzero_ps();

    if(matrix1==NULL||matrix2==NULL||ans==NULL){
        fprintf(stderr,"The matrixs are not valid! \n");//输出到错误流
        return false;
    }
    else if (matrix1->data == NULL || matrix2->data == NULL||ans->data==NULL) {
        fprintf(stderr,"The matrix data is not valid! \n");
        return false;
    } else if (matrix1->column != matrix2->row||matrix1->row != ans->row||matrix2->column!=ans->column) {
        return false;
    } else {
        float *p2 = (float *) malloc(sizeof(float) * matrix2->column * matrix1->column);
        if (p2 == NULL) {//申请空间失败
            fprintf(stderr,"p2 is failed to allocated\n");
            return false;
        }

        size_t o = 0;


        for (size_t j = 0; j < matrix2->column; ++j) {
// #pragma omp parallel for 
//             for (size_t i = 0; i < 8; i++)
//             {                           
                for (size_t k = 0; k < matrix2->column / 8 * 8; k += 8) {
                p2[o++] = matrix2->data[k * matrix2->column + j];
                p2[o++] = matrix2->data[(1 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(2 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(3 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(4 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(5 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(6 + k) * matrix2->column + j];
                p2[o++] = matrix2->data[(7 + k) * matrix2->column + j];
                }

            // }
            
            for (size_t k = matrix2->column / 8 * 8; k < matrix2->column; k++) {
                p2[o++] = matrix2->data[k * matrix2->column + j];

            }
        }


if(matrix1->row<8){
 float temp = 0;
        for (size_t i = 0; i < matrix1->row; ++i) {
            for (size_t j = 0; j < matrix2->column; ++j) {
                for (size_t k = 0; k < matrix1->column; ++k) {
                    temp += matrix1->data[i * matrix1->row + k] * matrix2->data[k * matrix1->column + j];
                }
                ans->data[i * matrix1->row + j] = temp;
                temp = 0;
            }
        }
        return true;
    }

else{

__m256 c[8] ;


#pragma omp parallel for 
for (size_t q = 0; q < 8; q++)
{
   __m256 c=_mm256_setzero_ps();

size_t m1row_8 = matrix1->row / 8;
float* matrix1_8 =  matrix1->data +( matrix1->row / 8 * q ) * matrix1->column ;
float* ans_8 = ans->data + (matrix1->row / 8 * q) * matrix2->column ;
        for (size_t i = 0; i < m1row_8; ++i) {
            for (size_t j = 0; j < matrix2->column; ++j) {
                float temp = 0;
                float sum[8] = {0};
                 c = _mm256_setzero_ps();
                ans_8[i * matrix1->row + j] = 0;

                size_t i1 = i * matrix1->column;
                size_t j1 = j * matrix2->row;

                for (size_t k = 0; k < matrix1->column / 8 * 8; k += 8) {
                    a = _mm256_loadu_ps(matrix1_8 + i1 + k);
                    b = _mm256_loadu_ps(p2 + j1 + k);
                    c = _mm256_add_ps(c, _mm256_mul_ps(a, b));

                }
                _mm256_storeu_ps(sum, c);

                ans_8[i * matrix2->column + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

                for (size_t k = matrix1->column / 8 * 8; k < matrix1->column; k += 1) {
                    //Tail case
                    temp += matrix1_8[i1 + k] * p2[j1+k];
                }
                ans_8[i * matrix2->column + j] += temp;
                temp = 0;
            }
            // printf("%d\n",i);
        }
}
 float* matrix1_else =  matrix1->data +( matrix1->row/ 8 * 8) * matrix1->column ;
float* ans_else = ans->data + (matrix1->row/ 8 * 8) * matrix2->column ;

        for (size_t i = 0; i < matrix1->row-matrix1->row /8* 8; ++i) {//tail
            for (size_t j = 0; j < matrix2->column; ++j) {
                float temp = 0;
                float sum[8] = {0};
                __m256 c = _mm256_setzero_ps();
                ans_else[i * matrix1->row + j] = 0;

                size_t i1 = i * matrix1->column;
                size_t j1 = j * matrix2->row;

                for (size_t k = 0; k < matrix1->column / 8 * 8; k += 8) {
                    a = _mm256_loadu_ps(matrix1_else + i1 + k);
                    b = _mm256_loadu_ps(p2 + j1 + k);
                    c = _mm256_add_ps(c, _mm256_mul_ps(a, b));

                }
                _mm256_storeu_ps(sum, c);

                ans_else[i * matrix2->column + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

                for (size_t k = matrix1->column / 8 * 8; k < matrix1->column; k += 1) {
                    //Tail case
                    temp+= matrix1_else[i1 + k] * p2[j1+k];                
}
                ans_else[i * matrix2->column + j] += temp;
                temp = 0;
            }
            // printf("%d\n",i);
        }


        free(p2);//释放空间
        return true;
    }
    }
// #else
//     printf("AVX2 is not supported");
// #endif

}


void initialMatrix(MatrixOne *matrix, const size_t row, const size_t col){//初始化一个一维矩阵
    if (row <= 0 || col <= 0) {
        fprintf(stderr,"invalid matrix, please put in again! \n");
    } else {
        matrix->row = row;
        matrix->column = col;
        matrix->data = (float *) malloc(sizeof(float) * row * col);//申请行空间
        if (matrix->data == NULL) {//申请空间失败
            fprintf(stderr,"The memory is failed to allocated\n");
        }
        // for (size_t j = 0; j < (unsigned long long) row * col; j++) {
        //     matrix->data[j] = 0;
        // }
    }
}


void createRamMatrix(MatrixOne *matrix, const size_t row, const size_t col, const size_t databound) {//检查合法性
    srand((unsigned) time(NULL));
    if (row <= 0||col<=0) {
        fprintf(stderr,"row or/and col is smaller than 0 \n");
    } else {
        matrix->row = row;
        matrix->column = col;
        matrix->data = (float *) malloc(sizeof(float) * row * col);//申请行空间
        // initialMatrixOneD(matrix, row, col);
        if (matrix->data == NULL) {//申请空间失败
            fprintf(stderr,"The memory is failed to allocated\n");
            free(matrix);
        }
        for (size_t j = 0; j < (unsigned long long) row * col; j++) {
            float ram = (float) rand() / RAND_MAX * databound;
            // float ram= rand()%2;
            matrix->data[j] = ram;
        }

    }
}


void showMatrix(const MatrixOne *matrix) {//打印矩阵
    if (matrix->data == NULL) {
        fprintf(stderr,"The matrix is null!");
    } else {
        for (size_t i = 0; i < matrix->row; i++) {
            for (size_t j = 0; j < matrix->column; j++) {
                printf("%-10.6f", matrix->data[i * matrix->row + j]);
            }
            printf("\n");
        }
    }
    printf("\n");

}