#ifndef UNTITLED_SOURCE_H
#define UNTITLED_SOURCE_H
typedef struct
{
    int row;
    int column;
    float **data; //便于为数据开辟内存
} Matrix;

typedef struct
{
    int row;
    int column;
    float *data;
} MatrixOne; // OneDimension

Matrix matmul_plain2D(const Matrix *matrix1, const Matrix *matrix2); //两个矩阵相乘
Matrix matmul_improved2D(const Matrix *matrix1, const Matrix *matrix2);

MatrixOne matmul_plain(const MatrixOne *matrix1, const MatrixOne *matrix2);    //矩阵相乘,一维
MatrixOne matmul_improved(const MatrixOne *matrix1, const MatrixOne *matrix2); //矩阵相乘,一维

void initialMatrix2D(Matrix *matrix, const int row, const int col);                               //初始化一个元素都为0的矩阵
void initialMatrixOneD(MatrixOne *matrix, const int row, const int col);                        //初始化一个元素都为0的矩阵,一维
void createRamMatrix(MatrixOne *matrix, const int row, const int col, const int databound); //检查合法性

void createRamMatrix2D(Matrix *matrix, const int row, const int col, const int databound); //检查合法性
void showMatrix2D(const Matrix *matrix);                                                   //打印矩阵
void showMatrix(const MatrixOne *matrix);                                             //打印矩阵,一维

#endif // UNTITLED_SOURCE_H
