#ifndef UNTITLED_SOURCE_H
#define UNTITLED_SOURCE_H
typedef struct {
    int row;
    int column;
    float **data; //便于为数据开辟内存
} Matrix;

typedef struct {
    int row;
    int column;
    float *data;
} MatrixOne;//OneDimension

Matrix matmul_plain(const Matrix *matrix1, const Matrix *matrix2); //两个矩阵相乘
Matrix matmul_improved(const Matrix *matrix1, const Matrix *matrix2);

Matrix matmul_improved11(const Matrix *matrix1, const Matrix *matrix2);

MatrixOne matmul_plainOneD(const MatrixOne *matrix1, const MatrixOne *matrix2);//矩阵相乘,一维
MatrixOne matmul_improvedOneD(const MatrixOne *matrix1, const MatrixOne *matrix2);//矩阵相乘,一维


void initialMatrix(Matrix *matrix, const int row, const int col);//初始化一个元素都为0的矩阵
void initialMatrixOne(MatrixOne *matrix, const int row, const int col); //初始化一个元素都为0的矩阵,一维
void createRamMatrixOne(MatrixOne *matrix, const int row, const int col, const int databound);//检查合法性


void createRamMatrix(Matrix *matrix, const int row, const int col, const int databound);//检查合法性
void showMatrix(const Matrix *matrix);//打印矩阵
void showMatrixOne(const MatrixOne *matrix);//打印矩阵,一维


#endif //UNTITLED_SOURCE_H
