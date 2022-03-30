
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

__global__
void what_is_my_id(int N, unsigned int* const block,
    unsigned int* const thread,
    unsigned int* const warp,
    unsigned int* const calc_thread,
    unsigned int* const completionTime)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_idx >= N) {
        return;
    }
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_thread[thread_idx] = thread_idx;
    completionTime[thread_idx] = clock();
}

void formatPrinter(const char* matName, int N, unsigned int* const mat) {
    printf("%s:", matName);
    for (int i = 0; i < N; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%d\t", mat[i]);
    }
    printf("\n\n");
}

void showResult(int N, unsigned int* const block,
    unsigned int* const thread,
    unsigned int* const warp,
    unsigned int* const calc_thread,
    unsigned int* const completionTime)
{
    const char* colors[5] = { "block", "thread_Id", "warp", "calc_thread", "completionTime" };
    formatPrinter(colors[0], N, block);
    formatPrinter(colors[1], N, thread);
    formatPrinter(colors[2], N, warp);
    formatPrinter(colors[3], N, calc_thread);
    formatPrinter(colors[4], N, completionTime);
}

int main()
{
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 1 << 6;
    const int limN = (N / 32 + 1) * 32;
    size_t size = N * sizeof(int);

    unsigned int* block;
    unsigned int* thread;
    unsigned int* warp;
    unsigned int* calc_thread;
    unsigned int* completionTime;

    cudaMallocManaged(&block, size);
    cudaMallocManaged(&thread, size);
    cudaMallocManaged(&warp, size);
    cudaMallocManaged(&calc_thread, size);
    cudaMallocManaged(&completionTime, size);

    cudaError_t asyncErr;

    what_is_my_id << <2, 37 >> > (N, block, thread, warp, calc_thread, completionTime);

    asyncErr = cudaDeviceSynchronize();
    if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    showResult(N, block, thread, warp, calc_thread, completionTime);

    cudaFree(block);
    cudaFree(thread);
    cudaFree(warp);
    cudaFree(calc_thread);
    cudaFree(completionTime);
}