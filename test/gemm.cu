#include <stdio.h>
#include <stdlib.h>

// A(M, N) @ B(N, K) = C(M, K)
__global__ void naive_gemm(float *A, float *B, float *C, int M, int N, int K) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < M && col < K) {
		float sum = 0.0f;
		for (int n=0; n<N; ++n)
			sum += A[row * N + n] * B[n * K + col];
		C[row * K + col]=sum;
	}
}

int main(void) {
	int M=1024, N=1024, K=1024;
	float *A, *B, *C, *dA, *dB, *dC;
	int asize = M * N * sizeof(float), bsize = N * K * sizeof(float), csize = M * K * sizeof(float);

	A = (float*)malloc(asize);
	B = (float*)malloc(bsize);
	C = (float*)malloc(csize);

	cudaMalloc((void**)&dA, asize);
	cudaMalloc((void**)&dB, bsize);
	cudaMalloc((void**)&dC, csize);

	cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bsize, cudaMemcpyHostToDevice);

	dim3 grid((M+15)/16, (K+15)/16);
	dim3 block(16,16);
	naive_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);

	cudaMemcpy(C, dC, csize, cudaMemcpyDeviceToHost);
}
