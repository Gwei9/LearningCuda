
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <crt/device_functions.h>
#include <corecrt_malloc.h>

#define imin(a,b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c)           // 点乘
{
    __shared__ float cache[threadsPerBlock];                // 用于存储每个线程的运行总和              
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;                           // 不需要将我们的块索引合并到这个偏移量中，因为每个块都有自己的这个共享内存的私有副本。

    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    // 在该块中同步线程
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // 将每个块的值存储到全局内存
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main()
{
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    // 在CPU端分配内存
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    // 错误处理
    cudaError_t cudaStatus;

    // 将内存分配到GPU上
    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMalloc failed!");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMalloc failed!");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMalloc failed!");
        return 1;
    }

    // 用数据填充主机内存
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 将数组' a '和' b '复制到GPU中
    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }
    cudaStatus = cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }

    dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

    // 将数组' c '从GPU复制到CPU
    cudaStatus = cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }

    c = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        c += partial_c[i];
    }

    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));
    
    // GPU端释放内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // CPU端释放内存
    free(a);
    free(b);
    free(partial_c);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
