#include <iostream>
#include <stdlib.h>
#include <time.h>


const int ROWS = 1024;
const int COLS = 1024;

using namespace std;

void matrix_mul_cpu(float* M, float* N, float* P, int width)
{
    for(int i=0;i<width;i++)
        for(int j=0;j<width;j++)
        {
            float sum = 0.0;
            for(int k=0;k<width;k++)
            {
                float a = M[i*width+k];
                float b = N[k*width+j];
                sum += a*b;
            }
            P[i*width+j] = sum;
        }
}

int main()
{
    clock_t start, end;
    start=clock();
    float *A, *B, *C;
    int total_size = ROWS*COLS*sizeof(float);
    A = (float*)malloc(total_size);
    B = (float*)malloc(total_size);
    C = (float*)malloc(total_size);

    //CPU一维数组初始化
    for(int i=0;i<ROWS*COLS;i++)
    {
        A[i] = 19.0;
        B[i] = 20.0;
    }

    matrix_mul_cpu(A, B, C, COLS);

    end=clock();
    cout << "total time is " << end-start << "ms" <<endl;

    return 0;
}
