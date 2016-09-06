#include <cstdio>
#include <iostream>
#include <time.h>

#include "PI.h"

using std::ofstream;
using std::cout;
using std::endl;

void solvePagerankByPICPU(float alpha, float eps, int num_n, int matrix_size, int* head, int* next, int* to, int* out_deg){
	float *weight;
	float *last_iteration_weight;
	
	weight = (float*)malloc(num_n * sizeof(float));
	last_iteration_weight = (float*)malloc(num_n * sizeof(float));

	for (int i = 0; i < num_n; i++){
		last_iteration_weight[i] = 1;
		weight[i] = 0;
	}
	int iteration_time = 0;
	bool need_next_iteration = true;

	cout << "running power iterations method at CPU..." << endl;
	clock_t start = clock();

	while (need_next_iteration){
		iteration_time++;
		need_next_iteration = false;
		for (int i = 0; i < num_n; i++){
			for (int j = head[i]; j != 0; j = next[j]){
				weight[to[j]] += alpha * last_iteration_weight[i] / out_deg[i];
			}
		}
		for (int i = 0; i < num_n; i++){
			weight[i] += 1 - alpha;
			if (fabs(weight[i] - last_iteration_weight[i]) > eps){
				need_next_iteration = true;
			}
			last_iteration_weight[i] = weight[i];
			weight[i] = 0;
		}
	}

	clock_t finish = clock();

	cout << "Number of Iterations:" << iteration_time << endl;
	float sum = 0;
	for (int i = 0; i < num_n; i++){
		sum += last_iteration_weight[i];
		//cout << i << ": " << last_iteration_weight[i] << endl;
	}
	cout << "total error: " << fabs(sum / num_n - 1) << endl;

	cout << "Elapsed Time: " << (double)(finish - start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
	cout << endl;

	free(last_iteration_weight);
	free(weight);
}