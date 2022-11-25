#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
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

    // clock_t start0, end0, start1, end1, start2, end2, start3, end3, start4, end4, start5, end5;

    size_t size = 8000;

    MatrixOne mm1;
    MatrixOne mm2;
    MatrixOne mm3;
    MatrixOne mm4;
    MatrixOne mm5;
    MatrixOne mm6;

    initialMatrix(&mm3,size,size);
    initialMatrix(&mm4,size,size);
    initialMatrix(&mm5,size,size);
    initialMatrix(&mm6,size,size);


    createRamMatrix(&mm1, size, size, 2);
    sleep(1);
    createRamMatrix(&mm2, size, size, 2);

    // showMatrix2D(&mm1);
    // showMatrix2D(&mm2);

    // mm4 = matmul_improved(&mm1, &mm1);//预热


    // start2 = clock();
    // makeBlock(size,&mm1, &mm2, &mm6);
    // end2 = clock();
    // printf("block spend time: %lf\n", (double) (end2 - start2) / CLOCKS_PER_SEC);
    // // showMatrix(&mm6);

    double start4 = omp_get_wtime();
    matmul_improved(&mm1, &mm2, &mm4);
    double end4 = omp_get_wtime();
    printf("improved spend time: %lf\n", (double) (end4 - start4));
    // showMatrix(&mm4);



    double start5=omp_get_wtime();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1, mm1.data, size, mm2.data, size, 0, mm5.data, size);
    double end5=omp_get_wtime();
    printf("Openblas spend time: %lf\n", (double) (end5 - start5) );
    // showMatrix(&mm5);




    float difference = 0;//看结果与OpenBlas的差异
    for (size_t i = 0; i < size*size; i++) {
        // printf("%f  ",fabsf(mm5.data[i] - mm4.data[i]));
        difference = difference + (fabsf(mm5.data[i] - mm4.data[i]) < 0.00001 ? 0 : fabsf(mm5.data[i] - mm4.data[i]));
    }
    printf("difference: %.6f\n", difference);


    // double start3=omp_get_wtime();
    // matmul_plain(&mm1, &mm2,&mm3);
    // double end3=omp_get_wtime();
    // printf("plainOneD spend time: %lf\n", (double) (end3 - start3));
    // // showMatrix(&mm3);

    // double start_1=omp_get_wtime();
    // matmul_plain11(&mm1, &mm2,&mm5);
    // double end_1=omp_get_wtime();
    // printf("plainOneD-1 spend time: %lf\n", (double) (end_1 - start_1));
    // // showMatrix(&mm5);


    // float difference = 0;//看结果与OpenBlas的差异
    // for (size_t i = 0; i < size*size; i++) {
    //     // printf("%f  ",fabsf(mm5.data[i] - mm4.data[i]));
    //     difference = difference + (fabsf(mm5.data[i] - mm3.data[i]) < 0.00001 ? 0 : fabsf(mm5.data[i] - mm3.data[i]));
    // }
    // printf("difference: %.6f\n", difference);

}