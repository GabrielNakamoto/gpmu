#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define N 256

__global__ void vector_mulpi(float *A, float *B) {
	A[threadIdx.x] = B[threadIdx.x] * 3.14f;
}

int main(void) {
	srand((unsigned int)time(NULL));

	float *A, *B, *d_A, *d_B;

	int size = N * sizeof(float);
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	for (int i=0; i<N; ++i) B[i]= 15.0 * ((float)rand() / (float)RAND_MAX);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	vector_mulpi<<<1,N>>>(d_A, d_B);

	cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
}
