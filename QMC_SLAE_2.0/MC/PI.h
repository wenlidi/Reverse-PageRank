#ifndef PI_H_
#define PI_H_

void solvePagerankByPICPU(float alpha, float eps, int num_n, int matrix_size, int* head, int* next, int* to, int* out_deg);

#endif