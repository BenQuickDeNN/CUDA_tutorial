/*********************************************************************
 * @file 	check_gpuinfo.cu
 * @brief 	display gpu information
 * @author 	Bin Qu
 * @email 	benquickdenn@foxmail.com
 * @date	2019-11-26
 * you can reedit or modify this file
*********************************************************************/

#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief display gpu information
 * @param devProp device
*/
void dispGPUInfo(const cudaDeviceProp& devProp);

/**
 * @brief fetch gpu information
 * @param dev_id gpu id
 * @return GPU information
*/
cudaDeviceProp getGPUInfo(const unsigned int& dev_id);

/**
 * @brief main entry
 * @return exit status
*/
int main(int argc, char** argv)
{
	// display the infomation of gpu0
	dispGPUInfo(getGPUInfo(0));

	return 0;
}

cudaDeviceProp getGPUInfo(const unsigned int& dev_id)
{
	std::printf("----------------GPU----------------\r\n");
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev_id);
	return devProp;
}
void dispGPUInfo(const cudaDeviceProp& devProp)
{
	std::printf("GPU name: %s\r\n", devProp.name);
	std::printf("number of SMs: %d\r\n", devProp.multiProcessorCount);
	std::printf("warp size: %d\r\n", devProp.warpSize);
	std::printf("max number of thread per SM: %d\r\n", devProp.maxThreadsPerMultiProcessor);
	std::printf("number of warp per SM: %d\r\n", devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
}