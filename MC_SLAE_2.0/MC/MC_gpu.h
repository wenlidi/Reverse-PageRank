#ifndef MC_GPU_H_
#define MC_GPU_H_

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

void solvePagerankByMCGPU(float alpha, int dimension, int num_m, int threads_per_block, int num_n, int matrix_size, unsigned int seed,
	float* &h_matrix_l, int* &h_column, int* &h_accu_row_size, float* &h_matrix_l_row_sum, int sims_per_thread);
#endif