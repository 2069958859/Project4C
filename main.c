#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <cblas.h>
#include <unistd.h>
#include <math.h>
//  g++ main.c source.c -mavx2 -DWITH_AVX2 && ./a.out
//sudo su -
//dd if=/dev/zero of=/swapfile bs=1048576 count=30720 开辟虚拟内存
//mkswap /swapfile
//swapon /swapfile

// cc -o mul main.c source.c -mavx2 -DWITH_AVX2 -I /usr/include/ -L/usr/lib -lopenblas -lpthread -lgfortran &&./mul
//cmake . -DCMAKE_BUILD_TYPE=Release -Bbuild
int main() {

    clock_t start0, end0, start1, end1, start2, end2, start3, end3, start4, end4, start5, end5;

    Matrix m1;
    int size = 1000;
    // createRamMatrix(&m1, size, size, 2);
    // Matrix m2;
    // createRamMatrix(&m2, size, size, 2);
    // Matrix m3;
    // Matrix m4;
    // Matrix m5;

    MatrixOne mm1;
    MatrixOne mm2;
    MatrixOne mm3;
    MatrixOne mm4;


    createRamMatrix(&mm1, size, size, 2);
    sleep(1);
    createRamMatrix(&mm2, size, size, 2);

    // showMatrix2D(&mm1);
    // showMatrix2D(&mm2);



//     m3 = matmul_plain(&m1, &m2);//预热

//     start0 = clock();
//     m3 = matmul_plain(&m1, &m2);
//     end0 = clock();
//     printf("plain spend time: %lf \n", (double) (end0 - start0) / CLOCKS_PER_SEC);
// //  showMatrix2D(&m3);


//     start1 = clock();
//     m4 = matmul_improved2D(&m1, &m2);
//     end1 = clock();
//     printf("improve spend time: %lf\n", (double) (end1 - start1) / CLOCKS_PER_SEC);
//     showMatrix2D(&m4);

    // mm4 = matmul_improved2D(&mm1, &mm1);//预热


    start4 = clock();
    mm4 = matmul_improved(&mm1, &mm2);
    end4 = clock();
    printf("improved spend time: %lf\n", (double) (end4 - start4) / CLOCKS_PER_SEC);
    // showMatrix2D(&mm4);


    MatrixOne mm5;

    start5 = clock();
    mm5.row = size;
    mm5.column = size;
    mm5.data = (float *) malloc(sizeof(float) * size * size);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1, mm1.data, size, mm2.data, size, 0, mm5.data, size);
    end5 = clock();
    printf("Openblas spend time: %lf\n", (double) (end5 - start5) / CLOCKS_PER_SEC);
    // showMatrix2D(&mm5);


    float difference = 0;//看结果与OpenBlas的差异
    for (size_t i = 0; i < size * size; i++) {
        difference += fabsf(mm5.data[i] - mm4.data[i]);
    }
    printf("difference: %.8f\n", difference);


    // start3 = clock();
    // mm3 = matmul_plain(&mm1, &mm1);
    // end3 = clock();
    // printf("plainOneD spend time: %lf\n", (double) (end3 - start3) / CLOCKS_PER_SEC);
    // showMatrix2D(&mm3);

    free(mm1.data);
    free(mm2.data);

}