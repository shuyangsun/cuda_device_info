# CUDA Device Information Printer
A CUDA program that prints the GPU information.

```
Sample Output:

------------------------------------------------------------------------------
                               CUDA Device Info                               
------------------------------------------------------------------------------
Detected 1 CUDA capable device

Device 0: "TITAN X (Pascal)"
	CUDA driver / runtime version: 8.0 / 8.0
	CUDA computing capability: 6.1
	Total amount of global memory: 11.90 GB (12778274816 bytes)
	GPU clock rate: 1.53 GHz (1531 MHz)
	Memory clock rate: 5.01 GHz (5005 MHz) 
	Memory bus width : 384-bit
	L2 cache size: 3.00 MB (3145728 bytes)
	Max texture dimension size (x, y, z):
		- 1D=(131072)
		- 2D=(131072, 65536)
		- 3D=(16384, 16384, 16384)
	Max layered texture size (dim) * layers:
		- 1D=(32768) * 2048
		- 2D=(32768, 32768) * 2048
	Total amount of constant memory: 64.00 KB (65536 bytes)
	Total amount of shared memory per block: 48.00 KB (49152 bytes)
	Total number of registers available per block: 65536
	Warp size: 32
	Multiprocessor count: 28
	Maximum number of threads per multiprocessor: 2048
	Maximum number of threads per block: 1024
	Maximum sizes of each dimension of a block: 1024 * 1024 * 64
	Maximum sizes of each dimension of a grid: 2147483647 * 65535 * 65535
	Maximum memory pitch: 2.00 GB (2147483647 bytes)
------------------------------------------------------------------------------
```
