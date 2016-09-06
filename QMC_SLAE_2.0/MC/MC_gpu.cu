#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <curand.h>
#include <algorithm>
#include<device_launch_parameters.h>
#include <numeric>
#include <time.h>
#include "MC_gpu.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::min;

__global__ void CalculateParametersGPU_Kernel(int num_n, float* d_matrix_p, float* d_matrix_l, int* d_column, int* d_accu_row_size, float* d_matrix_l_row_sum) {
	const unsigned int num_thread = blockDim.x;
	const unsigned int tid = blockIdx.x * num_thread + threadIdx.x;
	if (tid >= num_n){
		return ;
	}
	unsigned int begin = d_accu_row_size[tid];
	unsigned int end = d_accu_row_size[tid + 1];

	d_matrix_p[begin] = d_matrix_l[begin] / d_matrix_l_row_sum[tid];
	for (int i = begin + 1; i < end; i++){
		d_matrix_p[i] = d_matrix_l[i] / d_matrix_l_row_sum[tid] + d_matrix_p[i - 1];
	}
	return ;
}


void calculateMatrixpAndMatrixbGPU(int device_id, int num_n, float* &d_matrix_p, float* &d_matrix_l, int* &d_column, int* &d_accu_row_size, 
									float* &d_matrix_l_row_sum, float &f_value, float alpha){
	//don't need to calculate MatrixB
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	dim3 dim_grid, dim_block;
	dim_block.x = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
	if (dim_block.x >= num_n) {
		dim_block.x = num_n;
	}
	dim_grid.x = num_n / dim_block.x;
	if (num_n % dim_block.x != 0){
		dim_grid.x++;
	}

	f_value = (1 - alpha);

	/*
	//test
	cout <<"thread" << " " <<dim_block.x << " " << dim_grid.x << endl;
	cout << "f: " << f_value << endl;
	*/

	CalculateParametersGPU_Kernel << <dim_grid, dim_block >> >(num_n, d_matrix_p, d_matrix_l, d_column, d_accu_row_size, d_matrix_l_row_sum);
}

void generatePseudoRandomNumber(unsigned int seed, int d, int m, float* d_point){
	curandStatus_t curandResult;
	curandGenerator_t prng;
	curandResult = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	/*
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
		string msg("Could not create pseudo-random number generator: ");
		msg += curandResult;
		throw std::runtime_error(msg);
	}
	*/
	curandResult = curandSetPseudoRandomGeneratorSeed(prng, seed);
	/*
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
		string msg("Could not set seed for pseudo-random number generator: ");
		msg += curandResult;
		throw std::runtime_error(msg);
	}
	*/
	curandGenerateUniform(prng, (float *)d_point, d * m);

	/*
	if (typeid(Real) == typeid(float))
	{
		curandResult = curandGenerateUniform(prng, (float *)d_points, 2 * m_numSims);
	}
	else if (typeid(Real) == typeid(double))
	{
		curandResult = curandGenerateUniformDouble(prng, (double *)d_points, 2 * m_numSims);
	}
	else
	{
		string msg("Could not generate random numbers of specified type");
		throw std::runtime_error(msg);
	}
	*/
	/*
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
		string msg("Could not generate pseudo-random numbers: ");
		msg += curandResult;
		throw std::runtime_error(msg);
	}
	*/
	curandResult = curandDestroyGenerator(prng);
	/*
	if (curandResult != CURAND_STATUS_SUCCESS)
	{
		string msg("Could not destroy pseudo-random number generator: ");
		msg += curandResult;
		throw std::runtime_error(msg);
	}
	*/
}

__global__ void reduce_sum_array(float* d_result)	
{
	extern __shared__ float sdata[];

	// Perform first level of reduction:
	// - Write to shared memory
	unsigned int ltid = threadIdx.x;

	sdata[ltid] = d_result[ltid];
	__syncthreads();

	// Do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ltid < s)
		{
			sdata[ltid] += sdata[ltid + s];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		/*
		for (unsigned int s = blockDim.x; s > 1; s >>= 1){
			if (s % 2 == 1){
				sdata[0] += sdata[s - 1];
			}
		}
		*/
		d_result[0] = sdata[0];
	}
}

__device__ float reduce_sum(float in)
{
	extern __shared__ float sdata[];

	// Perform first level of reduction:
	// - Write to shared memory
	unsigned int ltid = threadIdx.x;

	sdata[ltid] = in;
	__syncthreads();

	// Do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ltid < s)
		{
			sdata[ltid] += sdata[ltid + s];
		}

		__syncthreads();
	}

	return sdata[0];
}

__global__ void RunMonteCarloGPUForOneVariable_Kernel(float* d_matrix_p, int* d_column, int* d_accu_row_size, float* d_matrix_l_row_sum,
	float f_value, float* d_point, int num_m, int dimension, int init_var, float* d_result, int sims_per_thread)
{
	// Determine thread ID
	unsigned int bid = blockIdx.x;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_m){
		return ;
	}
	
	float w = 1.0;
	float exp = 0;
	int cur_state = init_var;
	for (int t = 0; t < sims_per_thread; t++){
		w = 1.0;
		exp += f_value;
		cur_state = init_var;
		for (unsigned int i = 0; i < dimension; i++)
		{

			//float probability = d_point[t * dimension * num_m + tid + i * num_m];
			float probability = d_point[t * num_m + tid + i * num_m * sims_per_thread + 500];
			int l = d_accu_row_size[cur_state];
			int r = d_accu_row_size[cur_state + 1] - 1;
			if (l == r + 1) break;
			int mid;
			while (l < r) {
				mid = (l + r) / 2;
				if (d_matrix_p[mid] < probability){
					l = mid + 1;
				}
				else
				{
					r = mid;
				}
			}
			w = w * d_matrix_l_row_sum[cur_state];
			cur_state = d_column[l];  // Transfer to the next state.
			exp += w * f_value;  // w(k) * f(S(k)).

		}
	}
	exp = exp / sims_per_thread;
	// Reduce within the block
	__syncthreads();
	exp = reduce_sum(exp);

	// Store the result
	if (threadIdx.x == 0)
	{
		d_result[bid] = exp;
	}
	
}

void simulateMC(int device_id, int num_n, float* &d_matrix_p, int* &d_column, int* &d_accu_row_size, float* &d_matrix_l_row_sum, float f_value, 
	            float* &d_point, int &dimension, int &num_m, unsigned int seed, int threads_per_block, int sims_per_thread,
				float* &h_matrix_p, int* &h_column, int* &h_accu_row_size, float* &h_matrix_l_row_sum){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);
	dim3 dim_grid, dim_block;
	dim_block.x = min(prop.maxThreadsDim[0], prop.maxThreadsPerBlock);
	//
	dim_block.x = threads_per_block;
	//
	if (dim_block.x >= num_m) {
		dim_block.x = num_m;
	}
	dim_grid.x = num_m / dim_block.x;

	float *d_result;
	cudaMalloc((void **)&d_result, dim_grid.x * sizeof(float));
	vector<float> results(dim_grid.x);
	float *h_exp;
	h_exp = (float*)malloc(sizeof(float) * num_n);

	curandStatus_t curandResult;
	curandGenerator_t qrng;
	curandResult = curandCreateGenerator(&qrng, CURAND_RNG_QUASI_SOBOL32);

	curandResult = curandSetQuasiRandomGeneratorDimensions(qrng, dimension);

	curandResult = curandSetGeneratorOrdering(qrng, CURAND_ORDERING_QUASI_DEFAULT);

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);

	float generate_random_number_time;
	cudaThreadSynchronize();
	sdkStartTimer(&hTimer);
	curandResult = curandGenerateUniform(qrng, (float *)d_point, (500 + sims_per_thread * num_m) * dimension);

	cudaThreadSynchronize();
	sdkStopTimer(&hTimer);
	generate_random_number_time = sdkGetTimerValue(&hTimer);

	/*
	//test
	float *h_point;
	h_point = (float*)malloc(sizeof(float) * sims_per_thread * num_m * dimension);
	cudaMemcpy(h_point, d_point, sims_per_thread * num_m * dimension * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sims_per_thread * num_m * dimension; i++){
		
		cout << h_point[i] << " ";
		if (i % dimension == dimension - 1){
			cout << endl;
		}
		
		if (h_point[i] > 1){
			cout << "error" << endl;
			break;
		}
	}
	cout << endl << endl;
	cout << endl << endl;

	float exp = 0;
	float w = 1;
	for (int tid = 0; tid < 2; tid++){
		cout << "round:1" << endl;
		w = 1.0;
		exp += f_value;
		int cur_state = 8635;
		for (unsigned int i = 0; i < dimension; i++)
		{

			//float probability = d_point[t * dimension * num_m + tid + i * num_m];
			float probability = h_point[tid + i * num_m * sims_per_thread];
			cout << probability << endl;
			int l = h_accu_row_size[cur_state];
			int r = h_accu_row_size[cur_state + 1] - 1;
			cout << l << " " << r << endl;
			if (l == r + 1) break;
			int mid;
			while (l < r) {
				mid = (l + r) / 2;
				cout << h_matrix_p[mid] << endl;
				if (h_matrix_p[mid] < probability){
					l = mid + 1;
				}
				else
				{
					r = mid;
				}
			}
			cout << l << endl;
			w = w * h_matrix_l_row_sum[cur_state];
			cout << w << endl;
			cur_state = h_column[l];  // Transfer to the next state.
			exp += w * f_value;  // w(k) * f(S(k)).
			cout << exp << endl;
		}
	}
	*/
	
	cout << "Elapsed Time of generate random number per variable: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
	sdkResetTimer(&hTimer);

	float last_time = 0;
	float fastest_elapsed_time = 0;
	float slowest_elapsed_time = 0;

	for (int i = 0; i < num_n; i++){
		sdkStartTimer(&hTimer);
		RunMonteCarloGPUForOneVariable_Kernel << <dim_grid, dim_block, dim_block.x * sizeof(float)>> >(d_matrix_p, d_column, d_accu_row_size,
			d_matrix_l_row_sum, f_value, d_point, num_m, dimension, i, d_result, sims_per_thread);
		reduce_sum_array << <(1, 1, 1), dim_grid, dim_grid.x * sizeof(float) >> >(d_result);
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);

		if (fastest_elapsed_time == 0 || fastest_elapsed_time > sdkGetTimerValue(&hTimer) - last_time){
			fastest_elapsed_time = sdkGetTimerValue(&hTimer) - last_time;
		}
		if (slowest_elapsed_time == 0 || slowest_elapsed_time < sdkGetTimerValue(&hTimer) - last_time){
			slowest_elapsed_time = sdkGetTimerValue(&hTimer) - last_time;
		}
		last_time = sdkGetTimerValue(&hTimer);
		//cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
		cudaMemcpy(&h_exp[i], d_result, sizeof(float), cudaMemcpyDeviceToHost);
		//h_exp[i] = static_cast<float>(std::accumulate(results.begin(), results.end(), 0)) / num_m;
		h_exp[i] = h_exp[i] / num_m;
		//cout << "finish calculating " << i << "node's expectation" << endl;
		cout << h_exp[i] << endl;
		if (h_exp[i] > num_n){
			cout << i << "error" << endl;
			break;
		}
	}

	//curandDestroyGenerator(prng);

	cout << "Elapsed Time: " << sdkGetTimerValue(&hTimer) + generate_random_number_time<< "ms" << endl;
	cout << "average elapsed time for one variable: " << sdkGetTimerValue(&hTimer)  / num_n << "ms" << endl;
	cout << "minimal elapsed time for one variable£º" << fastest_elapsed_time<< "ms" << endl;
	cout << "maximal elapsed time for one variable£º" << slowest_elapsed_time<< "ms" << endl;
	cout << endl;

	float sum = 0;
	//cout << "ans:" << endl;
	for (int i = 0; i < num_n; i++){
		sum += h_exp[i];
	}
	//cout << endl;
	//cout << sum << endl;
	cout << "total error: " << fabs(sum  / num_n - 1) << endl;
	cout << endl;
}	

void solvePagerankByMCGPU(float alpha, int dimension, int num_m, int threads_per_block, int num_n, int matrix_size, unsigned int seed, 
	float* &h_matrix_l, int* &h_column, int* &h_accu_row_size, float* &h_matrix_l_row_sum, int sims_per_thread){
	int device_id = gpuGetMaxGflopsDeviceId();
	cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	cout << "running Quasi Metro Carlo method at GPU: ";
	cout << prop.name << endl;

	float *d_matrix_p;
	float *d_matrix_l;
	int *d_column;
	int *d_accu_row_size;
	float *d_matrix_l_row_sum;
	float f_value;

	cudaMalloc((void **)&d_matrix_p, matrix_size * sizeof(float));
	cudaMalloc((void **)&d_matrix_l, matrix_size * sizeof(float));
	cudaMalloc((void **)&d_column, matrix_size * sizeof(int));
	cudaMalloc((void **)&d_accu_row_size, (num_n + 1) * sizeof(int));
	cudaMalloc((void **)&d_matrix_l_row_sum, num_n * sizeof(float));

	cudaMemcpy(d_matrix_l, h_matrix_l, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_column, h_column, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_accu_row_size, h_accu_row_size, (num_n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_l_row_sum, h_matrix_l_row_sum, num_n * sizeof(float), cudaMemcpyHostToDevice);

	calculateMatrixpAndMatrixbGPU(device_id, num_n, d_matrix_p, d_matrix_l, d_column, d_accu_row_size, d_matrix_l_row_sum, f_value, alpha);

	//cout << "finish calculating matrixP" << endl;
	
	
	//test
	float *h_matrix_p;
	h_matrix_p = (float*)malloc(matrix_size * sizeof(float));
	cudaMemcpy(h_matrix_p, d_matrix_p, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
	/*
	cout << "matrixP" << endl;
	int row_index = 0;
	for (int i = 0; i < matrix_size; i++){
		if (h_accu_row_size[row_index] <= i){
			row_index++;
		}
		cout << row_index - 1 << " " << h_column[i] << " " << h_matrix_p[i] << endl;
	}
	//end of test
	*/

	float *d_point;
	cudaMalloc((void **)&d_point, sims_per_thread * dimension * num_m * sizeof(float));

	simulateMC(device_id, num_n, d_matrix_p, d_column, d_accu_row_size, d_matrix_l_row_sum, f_value, d_point, dimension, num_m, seed, threads_per_block, sims_per_thread
		, h_matrix_p, h_column, h_accu_row_size, h_matrix_l_row_sum);

	cudaDeviceReset();
}
