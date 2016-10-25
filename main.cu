
#include <iostream>
#include <cstdio>

#ifdef __CUDACC__

__host__
void CheckCUDAErr(const std::initializer_list<const cudaError_t>& errors);
__host__
void PrintGPUDeviceInfo();

#endif

int main(int argc, const char* argv[]) {

#ifdef __CUDACC__
	PrintGPUDeviceInfo();
#else
	fprintf(stderr, "Error: please use nvcc to compile the program.");
#endif
	return 0;
}


#ifdef __CUDACC__

__host__
void PrintGPUDeviceInfo() {
	printf("------------------------------------------------------------------------------\n");
	printf("                               CUDA Device Info                               \n");
	printf("------------------------------------------------------------------------------\n");
	int device_count {};
	CheckCUDAErr({cudaGetDeviceCount(&device_count)});

	if (device_count == 0) {
		fprintf(stderr, "No available CUDA device\n");
	} else {
		printf("Detected %u CUDA capable device%s\n", device_count, device_count <= 1 ? "" : "s");
		for (unsigned int device_idx = 0; device_idx < device_count; ++device_idx) {
			printf("\n");
			cudaSetDevice(device_idx);

			cudaDeviceProp device_prop {};
			int driver_version {};
			int runtime_version {};

			CheckCUDAErr({
				cudaGetDeviceProperties(&device_prop, device_idx),
				cudaDriverGetVersion(&driver_version),
				cudaRuntimeGetVersion(&runtime_version)
			});

			printf("Device %u: \"%s\"\n", device_idx, device_prop.name);
			printf("\tCUDA driver / runtime version: %d.%d / %d.%d\n",
					driver_version/1000, (driver_version%100)/10, runtime_version/1000, (runtime_version%100)/10);
			printf("\tCUDA computing capability: %d.%d\n", device_prop.major, device_prop.minor);
			printf("\tTotal amount of global memory: %.2f GB (%lu bytes)\n",
					device_prop.totalGlobalMem/std::pow(1024.0f, 3.0f), device_prop.totalGlobalMem);
			printf("\tGPU clock rate: %.2f GHz (%.0f MHz)\n",
					device_prop.clockRate * 1e-6f, device_prop.clockRate * 1e-3f);
			printf("\tMemory clock rate: %.2f GHz (%.0f MHz) \n",
					device_prop.memoryClockRate * 1e-6f, device_prop.memoryClockRate * 1e-3f);
			printf("\tMemory bus width : %d-bit\n",
					device_prop.memoryBusWidth);
			if (device_prop.l2CacheSize) {
				printf("\tL2 cache size: %.2f MB (%d bytes)\n",
						device_prop.l2CacheSize/std::pow(1024.0f, 2.0f), device_prop.l2CacheSize);
			}
			printf("\tMax texture dimension size (x, y, z):\n");
			printf(	"\t\t- 1D=(%d)\n"
					"\t\t- 2D=(%d, %d)\n"
					"\t\t- 3D=(%d, %d, %d)\n",
					device_prop.maxTexture1D,
					device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
					device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
			printf("\tMax layered texture size (dim) * layers:\n");
			printf(	"\t\t- 1D=(%d) * %d\n"
					"\t\t- 2D=(%d, %d) * %d\n",
					device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1],
					device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1], device_prop.maxTexture2DLayered[2]);
			printf("\tTotal amount of constant memory: %.2f KB (%lu bytes)\n",
					device_prop.totalConstMem/1024.0f, device_prop.totalConstMem);
			printf("\tTotal amount of shared memory per block: %.2f KB (%lu bytes)\n",
					device_prop.sharedMemPerBlock/1024.0f, device_prop.sharedMemPerBlock);
			printf("\tTotal number of registers available per block: %d\n",
					device_prop.regsPerBlock);
			printf("\tWarp size: %d\n",
					device_prop.warpSize);
			printf("\tSM (streaming multiprocessor) count: %d\n", device_prop.multiProcessorCount);
			printf("\tWarps per SM: %d\n", static_cast<int>(std::floor(device_prop.maxThreadsPerMultiProcessor/device_prop.warpSize)));
			printf("\tMaximum number of threads per SM: %d\n", device_prop.maxThreadsPerMultiProcessor);
			printf("\tMaximum number of threads per block: %d\n", device_prop.maxThreadsPerBlock);
			printf("\tMaximum number of threads total: %d\n", device_prop.multiProcessorCount * device_prop.maxThreadsPerMultiProcessor);
			printf("\tMaximum sizes of each dimension of a block: %d * %d * %d\n",
					device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
			printf("\tMaximum sizes of each dimension of a grid: %d * %d * %d\n",
					device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
			printf("\tMaximum memory pitch: %.2f GB (%lu bytes)\n",
					device_prop.memPitch/std::pow(1024.0f, 3.0f), device_prop.memPitch);
		}
	}
	printf("------------------------------------------------------------------------------\n");
}

__host__
void CheckCUDAErr(const std::initializer_list<const cudaError_t>& errors) {
	#pragma unroll
	for (auto err: errors) {
		if (err != cudaSuccess) {
			fprintf(stderr, "%s\n", cudaGetErrorString(err));
			throw std::runtime_error {"CUDA error encountered."};
		}
	}
}

#endif
