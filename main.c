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

int main() {
    size_t size = 2000;

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
        difference = difference + fabsf(mm5.data[i] - mm4.data[i]);
    }
    printf("difference of improved and openblas: %.6f\n", difference/(size*size));


    double start3=omp_get_wtime();
    matmul_plain(&mm1, &mm2,&mm3);
    double end3=omp_get_wtime();
    printf("plainOneD spend time: %lf\n", (double) (end3 - start3));
    // showMatrix(&mm3);


}