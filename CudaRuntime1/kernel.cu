
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#define Numbers 100
#include <iostream>
#include<cuda.h>
#include <stdio.h>
const char* message = "Hello World !\n";

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__global__ void demoKernal1() {
    //printf("Hello World !\n");
    printf("Hello World !\n");
}

__global__ void demoKernal2(int* a) {
    a[threadIdx.x] = threadIdx.x * threadIdx.x;

}

__global__ void demoAccessingDimensions(int count) {
    //int lineCount = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0 &&
        threadIdx.y == 0 && blockIdx.y == 0 &&
        threadIdx.z == 0 && blockIdx.z == 0)
    {
        printf("%d , %d , %d , %d , %d , %d \n", gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z);
        count = count + 1;
        //lineCount++;
        printf("%d\n", count);

    }

}
#define matRow 5
#define matCol 6


__global__ void demo2D(unsigned* mat) {

    unsigned id = threadIdx.x * blockDim. y + threadIdx.y;
    mat[id] = id;



}



int main()
{
    /* INTIAL PROG
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    printf("starting operation\n");
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    int i = 1;
    printf("Iteration %d\n", i);
    i++;
    cudaError_t cudaStatus1 = addWithCuda(c, a, b, arraySize);
    printf("Iteration %d\n", i);
    i++;
    cudaError_t cudaStatus2 = addWithCuda(c, a, b, arraySize);
    printf("Iteration %d\n", i);
    i++;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    */

    /*demoKernal1 <<<1, 20 >>> ();

    cudaDeviceSynchronize();
    printf("All done !\n");*/

    /*
    int arr1[Numbers];
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Numbers; i++)
    {
        arr1[i] = i * i;
        printf("Value is %d\n iteration is %d \n", arr1[i], i);
    }
    auto finishTime = std::chrono::high_resolution_clock::now();

    auto timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(finishTime - startTime);

    std::cout << "Total Time taken " << timeTaken.count() << " Micro seconds" << "\n";
    */

    /*

    int a[Numbers], * da;
    int i;
    cudaMalloc(&da, Numbers * sizeof(int));
    demoKernal2 << < 1, Numbers >> > (da);
    cudaMemcpy(a, da, Numbers * sizeof(int), cudaMemcpyDeviceToHost);
    auto startTimeCuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Numbers; i++)
    {
        printf("%d\n", a[i]);
    }
    auto finishTimeCuda = std::chrono::high_resolution_clock::now();
    auto timeTakenCuda = std::chrono::duration_cast<std::chrono::microseconds>(finishTimeCuda - startTimeCuda);
    std::cout << "Total Time taken by Cuda " << timeTakenCuda.count() << " Micro seconds" << "\n";
    
    */

    /*
    dim3 grid(2, 3, 4);
    dim3 block(5, 6, 7);
    int lineCount = 0;
    demoAccessingDimensions << <grid, block >> > (lineCount);
    cudaDeviceSynchronize();
    printf("%d", lineCount);
    */


    dim3 blockDim(matRow, matCol, 1);
    unsigned* matrix, * hMatrix;

    cudaMalloc(&matrix, matRow * matCol * sizeof(unsigned));
    hMatrix = (unsigned*)malloc(matRow * matCol * sizeof(unsigned));

    demo2D << <1, blockDim >> > (matrix);
    cudaMemcpy(hMatrix, matrix, matRow * matCol * sizeof(unsigned), cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < matRow; i++)
    {
        for (unsigned j = 0; j < matCol; j++)
        {
            printf("%2d", hMatrix[i * matCol + j]);

        }
        printf("\n");
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
