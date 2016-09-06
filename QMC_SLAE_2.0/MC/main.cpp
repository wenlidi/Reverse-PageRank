#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>

#include "PI.h"
#include "MC_gpu.h"
#include "parameters.h"

using std::string;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;

void printGraphInformation(int num_n, int matrix_size, int* in_deg, int* out_deg, int zero_num){

	int max_out_deg = out_deg[0];
	int min_out_deg = out_deg[0];
	int max_in_deg = in_deg[0];
	int min_in_deg = in_deg[0];
	for (int i = 1; i < num_n; i++){
		if (out_deg[i] > max_out_deg){
			max_out_deg = out_deg[i];
		}
		if (out_deg[i] < min_out_deg){
			min_out_deg = out_deg[i];
		}
		if (in_deg[i] > max_in_deg){
			max_in_deg = in_deg[i];
		}
		if (in_deg[i] < min_in_deg){
			min_in_deg = in_deg[i];
		}
	}
	cout << "Number of primitive graph's 0 out-degree nodes:" << zero_num << endl;
	cout << "Number of nodes:" << num_n << endl;
	cout << "Number of edges:" << matrix_size << endl;
	cout << "Number of maximun out-degree:" << max_out_deg << endl;
	cout << "Number of minimun out-degree:" << min_out_deg << endl;
	cout << "Number of maximun in-degree:" << max_in_deg << endl;
	cout << "Number of minimun in-degree:" << min_in_deg << endl;
	cout << endl;
}

void getInputFromFile(const string& filename, int &num_n, int &matrix_size, 
	float* &h_matrix_l, int* &h_column, int* &h_accu_row_size, float* &h_matrix_l_row_sum, 
	int* &head, int* &next, int* &to, int* &in_deg, int* &out_deg, int* &zero_row, int &zero_row_count){

	cout << "reading graph information from " << filename << endl;

	ifstream infile(filename.c_str(), std::ios::in);

	infile >> num_n >> matrix_size;
	int *tmpnext;
	int *tmpto;
	int *tmphead;

	tmphead = (int*)malloc(sizeof(int) * num_n);
	tmpnext = (int*)malloc(sizeof(int) * matrix_size + 5);
	tmpto = (int*)malloc(sizeof(int) * matrix_size + 5);
	zero_row = (int*)malloc(sizeof(int) * num_n); 
	out_deg = (int*)malloc(sizeof(int) * num_n);
	in_deg = (int*)malloc(sizeof(int) * num_n);

	zero_row_count = 0;
	memset(tmphead, 0, sizeof(int) * num_n);
	memset(out_deg, 0, sizeof(int) * num_n);
	memset(in_deg, 0, sizeof(int) * num_n);

	int count = 1;
	int vala, valb;
	for (int i = 0; i < matrix_size; i++){
		infile >> vala >> valb;
		vala -= 1;
		valb -= 1;
		out_deg[vala]++;
		in_deg[valb]++;
		tmpto[count] = vala;
		tmpnext[count] = tmphead[valb];
		tmphead[valb] = count++;
	}

	for (int i = 0; i < num_n; i++){
		if (out_deg[i] == 0){
			zero_row[zero_row_count++] = i;
			matrix_size += num_n;
			
			out_deg[i] = num_n;

		}
	}

	printGraphInformation(num_n, matrix_size, in_deg, out_deg, zero_row_count);
	//cout << zero_row_count << endl;

	head = (int*)malloc(sizeof(int) * num_n);
	next = (int*)malloc(sizeof(int) * matrix_size + 5);
	to = (int*)malloc(sizeof(int) * matrix_size + 5);

	memset(head, 0, sizeof(int) * num_n);

	h_matrix_l = (float*)malloc(sizeof(float) * matrix_size);
	h_column = (int*)malloc(sizeof(int) * matrix_size);
	h_accu_row_size = (int*)malloc(sizeof(int) * (num_n + 1));
	h_matrix_l_row_sum = (float*)malloc(sizeof(float) * num_n);

	count = 0;
	int listcount = 1;
	for (int i = 0; i < num_n; i++){
		h_matrix_l_row_sum[i] = 0;
		h_accu_row_size[i] = count;
		for (int j = tmphead[i]; j != 0; j = tmpnext[j]){
			h_matrix_l[count] = alpha * 1.0 / out_deg[tmpto[j]];
			h_matrix_l_row_sum[i] += h_matrix_l[count];
			h_column[count++] =tmpto[j];

			to[listcount] = i;
			next[listcount] = head[tmpto[j]];
			head[tmpto[j]] = listcount++;

		}
		for (int j = 0; j < zero_row_count; j++){
			h_matrix_l[count] = alpha * 1.0 / num_n;
			h_matrix_l_row_sum[i] += h_matrix_l[count];
			h_column[count++] = zero_row[j];

			to[listcount] = i;
			next[listcount] = head[zero_row[j]];
			head[zero_row[j]] = listcount++;

			in_deg[i]++;
		}
	}
	h_accu_row_size[num_n] = count;

	free(tmphead);
	free(tmpto);
	free(tmpnext);
}

void test(const string& filename, int &num_n, int &matrix_size,
	float* &h_matrix_l, int* &h_column, int* &h_accu_row_size, float* &h_matrix_l_row_sum,
	int* &head, int* &next, int* &to, int* &in_deg, int* &out_deg, int* &zero_row, int &zero_row_count){
	cout << filename << endl;
	cout << num_n << endl;
	cout << matrix_size << endl;

	//test MatrixS
	cout << "matrixS" << endl;
	int row_index = 0;
	for (int i = 0; i < matrix_size; i++){
		if (h_accu_row_size[row_index] <= i){
			row_index++;
		}
		cout << row_index - 1 << " " << h_column[i] << " " << h_matrix_l[i] << endl;
	}
	cout << "row_sum" << endl;
	for (int i = 0; i < num_n; i++){
		cout << h_matrix_l_row_sum[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < num_n + 1; i++){
		cout << h_accu_row_size[i] << " ";
	}
	cout << endl;

	cout << "list:" << endl;
	for (int i = 0; i < num_n; i++){
		cout << i << ": ";
		for (int j = head[i]; j != 0; j = next[j]){
			cout << to[j] << " ";
		}
		cout << endl;
	}
}

void setParameters(){
	cout << "reading parameters from parameters.txt..." << endl;

	freopen("parameters.txt", "r", stdin);

	scanf("alpha = %f\n", &alpha);
	scanf("eps = %f\n", &eps);
	scanf("dimension = %d\n", &dimension);
	scanf("num_m = %d\n", &num_m);
	scanf("threads_per_block = %d\n", &threads_per_block);
	scanf("seed = %d\n", &seed);
	scanf("sims_per_thread = %d\n", &sims_per_thread);
	fclose(stdin);
	//test

	cout << "alpha = " << alpha << endl;
	cout << "eps = " << eps << endl;
	cout << "d = " << dimension << endl;
	cout << "num_m = " << num_m << endl;
	cout << "threads_per_block = " << threads_per_block << endl;
	cout << "seed = " << seed << endl;
	cout << "sims_per_thread = " << sims_per_thread << endl;
	cout << endl;
}

int main(int argc, char** argv){
	string input_file = "input.txt";
	string output_file = "output.txt";
	
	if (argc > 1){
		input_file = argv[1];
	}
	if (argc > 2){
		output_file = argv[2];
	}
	freopen(output_file.c_str(), "w", stdout);

	int num_n;
	int matrix_size;
	float* h_matrix_l;
	int* h_column;
	int* h_accu_row_size;
	float* h_matrix_l_row_sum;
	int* head;
	int* next;
	int* to;
	int* in_deg;
	int* out_deg;
	int* zero_row;
	int zero_row_count;
	setParameters();

	getInputFromFile(input_file.c_str(), num_n, matrix_size, h_matrix_l, h_column, h_accu_row_size, h_matrix_l_row_sum,
		head, next, to, in_deg, out_deg, zero_row, zero_row_count);

	//test(input_file.c_str(), num_n, matrix_size, h_matrix_l, h_column, h_accu_row_size, h_matrix_l_row_sum, head, next, to, in_deg, out_deg, zero_row, zero_row_count);
	
	solvePagerankByMCGPU(alpha, dimension, num_m, threads_per_block, num_n, matrix_size, seed, h_matrix_l, h_column, h_accu_row_size, h_matrix_l_row_sum, sims_per_thread);

	//solvePagerankByPICPU(alpha, eps, num_n, matrix_size, head, next, to, out_deg);

	fclose(stdout);
	
	return 0;
}