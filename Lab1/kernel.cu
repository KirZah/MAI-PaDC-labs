// Складываем два масссива
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

// if N >= 65 535 then error  // попробовал, ошибки почему-то нет!!!
#define N 6000
#define SEPARATOR '\n'


void print_cpu(int n, char separator) {
	int i = 1;
	while (i < n) {
		printf("%d%c", i, separator);
		i++;
	}
}

double count_print_time_using_cpu(int n, char separator) {
	clock_t begin = clock();
	/* here, do your time-consuming job */
	print_cpu(n, separator);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Calculation time using CPU: %.2f seconds\n", time_spent);

	//print_array_int(c, N);
	return time_spent;
}


__global__ void print_gpu(char separator) {
	printf("%d%c", blockIdx.x + 1, separator); 
}

double count_print_time_using_gpu(int n, char separator) {
	clock_t begin = clock();
	print_gpu<<<n, 1>>>(separator); // 65535 блоков, 1 нить
	cudaDeviceReset();
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


int main(void) {
	//printf("\nPress 'Enter' to write numbers using CPU..\n");
	//getchar();
	double cpu_printing_time = count_print_time_using_cpu(N, SEPARATOR);
	printf("\nPress 'Enter' to write numbers using GPU..\n");
	//getchar();
	double gpu_printing_time = count_print_time_using_gpu(N, SEPARATOR);
	printf("\n\n----------------------------------------------------\n Parameters:\n\tN = %d\n\tSEPARATOR = '%c'\n-------------\n Results:\n", N, SEPARATOR);
	printf("\tPrinting time using CPU took %.2f seconds\n", cpu_printing_time);
	printf("\tPrinting time using GPU took %.2f seconds\n----------------------------------------------------\n", gpu_printing_time);
	printf("\nWow!!\n");
}

