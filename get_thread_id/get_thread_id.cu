/*********************************************************************
 * @file 	check_gpuinfo.cu
 * @brief 	fetch thread ID
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2019-12-1
 * you can reedit or modify this file
*********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels.h"

/**
 * @brief display thread ID
 */
void dispThreadID();

/**
 * @brief main entry
 * @return exit status
 */
int main(int argc, char** argv)
{
	dispThreadID();
	return 0;
}

void dispThreadID()
{
	/* initialize grid */
	dim3 gridSize(2, 3, 4);

	/* initialize block */
	dim3 blockSize(5, 6, 7);

	/* allocate memory on host */
	const unsigned int memSpace = gridSize.z * gridSize.y * gridSize.x *
		blockSize.z * blockSize.y * blockSize.x;
	unsigned int* threadID;
	threadID = (unsigned int*)std::malloc(memSpace * sizeof(unsigned int));
	
	/* allocate memory on device */
	unsigned int* cuThreadId;
	cudaMalloc((void**)&cuThreadId, memSpace * sizeof(unsigned int));

	/* copy data from host to device */
	/* in this application, there is no need to copy data from host to device */
	//cudaMemcpy(cuThreadId, threadID, memSpce * sizeof(unsigned int), cudaMemcpyHostToDevice)

	/* call kernel */
	computeThreadID<<<gridSize, blockSize>>>(cuThreadId);

	/* copy data from device to host */
	cudaMemcpy(threadID, cuThreadId, memSpace * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(cuThreadId);

	/* display thread id */
	for (int i = 0; i < memSpace; i++)
		std::printf("%d, ", threadID[i]);
	std::printf("\r\n");

	/* free memory on host */
	std::free(threadID);
}