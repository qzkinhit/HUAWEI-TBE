
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h> 
#include <stdio.h>
#include <math.h>

const int Row = 1024;
const int Col = 1024;

__global__
void matrix_mul_gpu(float* M, float* N, float* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.0;
    for (int k = 0; k < width; k++)
    {
        float a = M[j * width + k];
        float b = N[k * width + i];
        sum += a * b;
    }
    P[j * width + i] = sum;
}

int main()
{
 
    clock_t start, end;
    start= clock();
    float* A = (float*)malloc(sizeof(float) * Row * Col);
    float* B = (float*)malloc(sizeof(float) * Row * Col);
    float* C = (float*)malloc(sizeof(float) * Row * Col);
    //malloc device memory
    float* d_dataA, * d_dataB, * d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(float) * Row * Col);
    cudaMalloc((void**)&d_dataB, sizeof(float) * Row * Col);
    cudaMalloc((void**)&d_dataC, sizeof(float) * Row * Col);
    //set value
    for (int i = 0; i < Row * Col; i++) {
        A[i] = 19.0;
        B[i] = 20.0;
    }

    cudaMemcpy(d_dataA, A, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(14, 14);
    dim3 blockNumber((Col + threadPerBlock.x - 1) / threadPerBlock.x, (Row + threadPerBlock.y - 1) / threadPerBlock.y);
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    //拷贝计算数据-一级数据指针
    cudaMemcpy(C, d_dataC, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost);

    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    end = clock();
    int timeuse =(end - start);
    printf("total time is %d ms\n", timeuse);

    return 0;
}