#include <stdio.h>
#include <malloc.h>
#include "source.h"
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
// g++ main.c source.c -mavx2 -DWITH_AVX2 && ./a.out


int main() {

    clock_t start1, end1,start0, end0;

    Matrix m1;
    int size = 17;
    createRamMatrix(&m1, size, size, 2);
    Matrix m2;
    createRamMatrix(&m2, size, size, 2);
    Matrix m3;
    Matrix m4;
//     showMatrix(&m1);
// showMatrix(&m2);


   start0=clock();
   m3=matmul_plain(&m1, &m2);
   end0=clock();
   printf("plain spend time: %lf \n", (double) (end0 - start0) / CLOCKS_PER_SEC);
 showMatrix(&m3);

   start1=clock();
   m4=matmul_improved(&m1, &m2);
   end1=clock();
   printf("improve spend time: %lf\n", (double) (end1 - start1) / CLOCKS_PER_SEC);
 showMatrix(&m4);


}