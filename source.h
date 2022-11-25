#include <stdbool.h>

#ifndef UNTITLED_SOURCE_H
#define UNTITLED_SOURCE_H
typedef struct
{
    size_t row;
    size_t column;
    float *data;
} MatrixOne; // OneDimension

// Matrix matmul_plain2D(const Matrix *matrix1, const Matrix *matrix2); //两个矩阵相乘
// Matrix matmul_improved2D(const Matrix *matrix1, const Matrix *matrix2);
// void initialMatrix2D(Matrix *matrix, const size_t row, const size_t col);                               //初始化一个元素都为0的矩阵
// void createRamMatrix2D(Matrix *matrix, const size_t row, const size_t col, const size_t databound); //检查合法性
// void showMatrix2D(const Matrix *matrix);                                                   //打印矩阵

// bool makeBlock(size_t n,const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans);//矩阵相乘,一维
// bool mulBlock(size_t n , const float *matrix1, const float *matrix2, float *ans);

bool matmul_plain(const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans);//矩阵相乘,一维
bool matmul_improved(const MatrixOne *matrix1, const MatrixOne *matrix2, MatrixOne *ans);//矩阵相乘,一维

void initialMatrix(MatrixOne *matrix, const size_t row, const size_t col);     //初始化一个一维矩阵
void createRamMatrix(MatrixOne *matrix, const size_t row, const size_t col, const size_t databound); //检查合法性

void showMatrix(const MatrixOne *matrix);                                             //打印矩阵,一维

#endif // UNTITLED_SOURCE_H
