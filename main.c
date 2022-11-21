#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
//  g++ main.c source.c -mavx2 -DWITH_AVX2 && ./a.out
//sudo su -
//dd if=/dev/zero of=/swapfile bs=1048576 count=30720
//mkswap /swapfile1
//swapon /swapfile1

int main() {

    clock_t start0, end0, start1, end1,  start2, end2,start3, end3,start4, end4;

    Matrix m1;
    int size = 16;
    int sizelarge = 16;
   createRamMatrix(&m1, size, size, 2);
   Matrix m2;
   createRamMatrix(&m2, size, size, 2);
   Matrix m3;
   Matrix m4;
   Matrix m5;
//
    MatrixOne mm1;
    MatrixOne mm2;
    MatrixOne mm3;
    MatrixOne mm4;


    createRamMatrixOne(&mm1, sizelarge, sizelarge, 2);

    //createRamMatrixOne(&mm2, sizelarge, sizelarge, 2);
//    showMatrixOne(&mm1);
//    showMatrixOne(&mm2);

//     showMatrix(&m1);
// showMatrix(&m2);


   start0=clock();
   m3=matmul_plain(&m1, &m2);
   end0=clock();
   printf("plain spend time: %lf \n", (double) (end0 - start0) / CLOCKS_PER_SEC);
//  showMatrix(&m3);


   start1=clock();
   m4=matmul_improved(&m1, &m2);
   end1=clock();
   printf("improve spend time: %lf\n", (double) (end1 - start1) / CLOCKS_PER_SEC);
//  showMatrix(&m4);

   start2=clock();
   m5=matmul_improved11(&m1, &m2);
   end2=clock();
   printf("improve11 spend time: %lf\n", (double) (end2 - start2) / CLOCKS_PER_SEC);

    start3 = clock();
    mm3 = matmul_plainOneD(&mm1, &mm1);
    end3 = clock();
    printf("plainOneD spend time: %lf\n", (double) (end3 - start3) / CLOCKS_PER_SEC);
  //  showMatrixOne(&mm3);


    start4 = clock();
    mm4 = matmul_improvedOneD(&mm1, &mm1);
    end4 = clock();
    printf("improvedOneD spend time: %lf\n", (double) (end4 - start4) / CLOCKS_PER_SEC);
  //  showMatrixOne(&mm4);





}