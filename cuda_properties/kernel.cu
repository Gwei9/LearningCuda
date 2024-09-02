
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


int main()
{
    cudaDeviceProp prop;

    int count;

    cudaError_t cudaStatus = cudaGetDeviceCount(&count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "getDeviceCount failed!");
    }

    for (int i = 0; i < count; ++i) {
        cudaStatus = cudaGetDeviceProperties(&prop, i);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "getDeviceProperties failed!");
        }

        printf("  --- General Information for device %d ---\n", i);
        printf("Name:  %s\n", prop.name);
        printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
        printf("Device copy overlap:  ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("  --- Memory Information for device %d ---\n", i);
        printf("Total global Mem: %1d\n", prop.totalGlobalMem);
        printf("Total constant Mem: %1d\n", prop.totalConstMem);
        printf("\n");
    }

    return 0;
}
