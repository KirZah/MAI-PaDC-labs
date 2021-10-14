#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// if N >= 1024 then incorrect (cause of memory limits)
#define N 512  // 32 * x

//using namespace std;


// returns array of solutions
/*
float* find_equasion_solutions_gpu(function, border_left, width, pos_x) {
	// Определить какой промежуток мы используем
	// (и точку в которой считаем значение функции)
	float thread_x = border_left + (blockIdx.x + pos_x) * width;

	// count function value
	// (may be bettered by analysing the equasion or by using less float values)


	// compare resuslt with 0

}
//*/


float calculate_function_cpu(float a, float b, float x) {
	return a + b * x;
}

__device__ float calculate_function_gpu(float a, float b, float x) {
	return a + b * x;
}


/**
* FIXME: should return the amount of solutions found:
*    0 if solution is not found
*	-1 if error input
*	// -2 if found solution, but slightly outside the borders
*/
float* find_solution_borders_cpu(float* borders_ptr, float a, float b,
		float border_left, float border_right, float width, float pos_x) {
	///printf("\t==========================\n");
	///printf("\tfind_solution_borders_cpu:\n");
	///printf("\t--------------------------\n");
	int intervals_n = (int)((border_right - border_left) / width);
	///printf("\t[%f, %f] - borders\n", border_left, border_right);
	///printf("\twidth = %f\n", width);
	///printf("\tintervals_n = %d\n", intervals_n);
	if (intervals_n < 2) {
		printf("\tERROR: intervals_n = %d\n", intervals_n);
		system("pause");
		exit(1);
	}
	if ((border_right - border_left) < width * 2) {
		printf("\tERROR: incorrect width;  intervals_n = %d\n", intervals_n);
		system("pause");
		exit(1);
	}

	float f_x, f_x_, x, x_;

	bool solution_is_found = false;
	// Определить какой промежуток мы используем 
	// (и точку в которой считаем значение функции для каждой нити)
	int idx = 0;
	for (; idx < intervals_n+1; idx++) { // intervals_n + 1 is because: |"---"---"-|
		x = border_left + (idx + pos_x) * width;  // for cpu

		// count function value 
		// (may be bettered by analysing the equasion or by using less float values)
		f_x = calculate_function_cpu(a, b, x);
		if (idx == 0) {
			f_x_ = f_x;
		}

		// compare resuslt with 0 (finish cycle if soulution is found)
		if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
			solution_is_found = true;
			///printf("\tFOUND solution in the %d interval\n", idx);
			break;
		}

		//remember if prev was less or more than 0
		x_ = x;
		f_x_ = f_x;
	}

	if (!solution_is_found) { // FIXME?
		///printf("\tSOLUTION IS NOT FOUND! checking if there's solution is beeween [x, right_border]\n");
		x = border_right;
		f_x = calculate_function_cpu(a, b, x);
		if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
			///printf("\tSolution IS FOUND in the LAST GAP!\n");
		}
	}

	borders_ptr[0] = x_;
	borders_ptr[1] = x;

	///printf("\tf(x) in [%f, %f]\n", f_x_, f_x);
	///printf("\tx    in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	//getchar();

	///printf("\t==========================\n\n");
	return borders_ptr;
}

float find_solution_cpu(float a, float b, float border_left,
		float border_right, float pos_x, float eps, int intervals_n) {
	printf("=======================\n");
	printf("count_solution_time_cpu\n");
	printf("-----------------------\n");

	// Dynamically allocate memory using malloc()
	float* borders_ptr = (float*)malloc(2 * sizeof(float));
	if (borders_ptr == NULL) {
		printf("ooooooops\n");
		getchar();
		exit(1);
	}
	borders_ptr[0] = border_left;
	borders_ptr[1] = border_right;


	float radius = border_right - border_left;
	float width = radius / intervals_n;

	printf("[%f, %f] - borders\n", borders_ptr[0], borders_ptr[1]);
	printf("radius = %f, \t", radius);
	printf("width  = %f\n", width);
	eps *= 2;
	printf("eps = %f\n", eps);

	int iterations = 0;
	while (radius > eps) {
		++iterations;
		borders_ptr = find_solution_borders_cpu(borders_ptr,
			a, b, borders_ptr[0], borders_ptr[1], width, pos_x);
		printf("RESULT %d: x in [%f, %f]\n", iterations, borders_ptr[0], borders_ptr[1]);

		radius = borders_ptr[1] - borders_ptr[0]; // abs() suppports only integer values!
		width = radius / intervals_n;

		///printf("radius = %f\t", radius);
		///printf("width  = %f\n\n", width);
		//system("pause");
	}
	printf("=======================\n");
	printf("FINAL RESULT: x in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	float result = borders_ptr[0] + (borders_ptr[1] - borders_ptr[0]) / 2;
	eps /= 2;
	printf("x = %f +- %f\n", result, eps);
	printf("Iterations = %d\n\n", iterations);

	printf("radius = %f\n", radius);
	printf("eps    = %f\n", eps);
	printf("=======================\n\n");
	free(borders_ptr);
	return result;
}

double count_solution_time_cpu(float a, float b, float border_left,
		float border_right, float pos_x, float eps, int intervals_n) {
	clock_t begin = clock();
	float result = \
		find_solution_cpu(a, b, border_left, border_right, pos_x, eps, intervals_n);
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


// LINEAR PART IS ON GPU

__device__ void find_solution_borders_gpu_device(float* borders_ptr, float a, float b,
	float border_left, float border_right, float width, float pos_x) {
	/*
	printf("blockIdx.x = %d,\t threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
	//float x = border_left + (blockIdx.x + pos_x) * width;  //for gpu
	//printf("res = %f\n", calculate_function_gpu(a, b, x));
	//*/

	printf("\t==========================\n");
	printf("\tfind_solution_borders_cpu:\n");
	printf("\t--------------------------\n");
	printf("\t[%f, %f] - borders\n", border_left, border_right);
	printf("\twidth = %f\n", width);
	int intervals_n = (int)((border_right - border_left) / width);
	printf("\tintervals_n = %d\n", intervals_n);
	if (intervals_n < 2) {
		printf("\tERROR: intervals_n = %d\n", intervals_n);
		//system("pause");
		//exit(1);
	}
	if ((border_right - border_left) < width * 2) {
		printf("\tERROR: incorrect width;  intervals_n = %d\n", intervals_n);
		//system("pause");
		//exit(1);
	}

	/////////////
	// FIXME: NEED TO CHANGE LAST GAP

	/// get prev: was less or more than 0?
	float x_ = border_left + ((threadIdx.x) + pos_x) * width,
		f_x_ = calculate_function_gpu(a, b, x_);


	// count function value 
	float x = border_left + ((threadIdx.x + 1) + pos_x) * width;  //for gpu
	// intervals_n + 1 is because: |"---"---"-|
	float f_x = calculate_function_gpu(a, b, x);

	// compare result with 0 (finish cycle if soulution is found)
	if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
		//solution_is_found = true;
		printf("\tFOUND solution in the %d thread!\n", threadIdx.x);

		borders_ptr[0] = x_;
		borders_ptr[1] = x;
		printf("\tf(x) in [%f, %f]\n", f_x_, f_x);
		printf("\tx    in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	}

	// FIXME (get rid of gap cheking by making search a bit wider)
	// Checking te GAP
	/*
	printf("\tChecking if there's solution beeween [x, right_border]\n");

	x = border_right;  //for gpu
	f_x = calculate_function_gpu(a, b, x);
		
	if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
		printf("\tSolution IS FOUND in the LAST GAP!\n");
	}
	//*/


	//////////////
	
	printf("\t==========================\n\n");
	//return borders_ptr;
}


__global__ void find_solution_gpu_on_device(float a, float b,
	float border_left, float border_right,
	float pos_x, float eps, int intervals_n) {
	/*
	//__syncthreads();
	cudaDeviceSynchronize();
	//printf("blockIdx.x = %d\n", blockIdx.x);
	__shared__

	find_solution_borders_gpu(float* borders_ptr, float a, float b,
			float border_left, float border_right, float width, float pos_x)
	//*/
	printf("=======================\n");
	printf("find_solution_gpu_on_device\n");
	printf("-----------------------\n");

	// Dynamically allocate memory using malloc()
	float* borders_ptr = (float*)malloc(2 * sizeof(float));
	if (borders_ptr == NULL) {
		printf("ERROR: ooooooops\n");
		//getchar();
		//exit(1);
	}
	borders_ptr[0] = border_left;
	borders_ptr[1] = border_right;


	float radius = border_right - border_left;
	float width = radius / intervals_n;

	printf("[%f, %f] - borders\n", borders_ptr[0], borders_ptr[1]);
	printf("radius = %f, \t", radius);
	printf("width  = %f\n", width);
	eps *= 2;
	printf("eps = %f\n", eps);

	int iterations = 0;
	while (radius > eps) {
		++iterations;
		// <<<1, intervals_n>>>
		find_solution_borders_gpu_device(borders_ptr,
			a, b, borders_ptr[0], borders_ptr[1], width, pos_x);

		__syncthreads();

		printf("RESULT : x in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);

		radius = borders_ptr[1] - borders_ptr[0]; // abs() suppports only integer values!
		width = radius / intervals_n;

		printf("radius = %f\t", radius);
		printf("width  = %f\n\n", width);
		//system("pause");
	}
	printf("=======================\n");
	printf("FINAL RESULT: x in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	float result = borders_ptr[0] + (borders_ptr[1] - borders_ptr[0]) / 2;
	eps /= 2;
	printf("x = %f +- %f\n", result, eps);
	printf("Iterations = %d\n\n", iterations);

	printf("radius = %f\n", radius);
	printf("eps    = %f\n", eps);
	printf("=======================\n\n");
	free(borders_ptr);
}


double count_solution_time_gpu_device(float a, float b,
	float border_left, float border_right,
	float pos_x, float eps, int intervals_n) {
	clock_t begin = clock();

	int solutiouns_n = 1;	//AMOUNT OF ABLE SOLUTIONS
	float* result = new float[solutiouns_n];
	float* result_dev = NULL;
	cudaMalloc((void**)&result_dev, solutiouns_n * sizeof(float));

	//dim3 threads = dim3(32, 1, 1);
	//dim3 blocks  = dim3(intervals_n / threads.x, 1, 1);



	// <<<1, 1>>>
	find_solution_gpu_on_device<<<1, intervals_n>>> (a, b,
		border_left, border_right,
		pos_x, eps, intervals_n
		);

	//system("pause");
	cudaMemcpy(&result, result_dev, solutiouns_n * sizeof(float), cudaMemcpyDeviceToHost);

	//system("pause");
	printf("--------------------------");
	printf("Result = %f\n", &result[0]);
	cudaFree(result_dev);

	free(result);

	//cudaDeviceReset();
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


// LINEAR PART IS ON CPU

__global__ void find_solution_borders_gpu_global(float* borders_ptr_dev, float a, float b,
	float border_left, float border_right, float width, float pos_x) {
	/*
	printf("blockIdx.x = %d,\t threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
	//float x = border_left + (blockIdx.x + pos_x) * width;  //for gpu
	//printf("res = %f\n", calculate_function_gpu(a, b, x));
	//*/

	int intervals_n = (int)((border_right - border_left) / width);
	if (intervals_n < 2) {
		printf("\tthreadIdx.x = %d: \tERROR: intervals_n = %d\n", threadIdx.x, intervals_n);
		//system("pause");
		//exit(1);
	}
	if ((border_right - border_left) < width * 2) {
		printf("\tthreadIdx.x = %d: \tERROR: incorrect width;  intervals_n = %d\n", threadIdx.x, intervals_n);
		//system("pause");
		//exit(1);
	}

	__syncthreads();
	/////////////
	// FIXME: NEED TO CHANGE LAST GAP

	/// get prev: was less or more than 0?
	float x_ = border_left + ((threadIdx.x) + pos_x) * width,
		f_x_ = calculate_function_gpu(a, b, x_);


	// count function value 
	float x = border_left + ((threadIdx.x + 1) + pos_x) * width;  //for gpu
	// intervals_n + 1 is because: |"---"---"-|
	float f_x = calculate_function_gpu(a, b, x);

	// compare result with 0 (finish cycle if soulution is found)
	if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
		///printf("\tthreadIdx.x = %d: \t==========================\n", threadIdx.x);
		///printf("\tthreadIdx.x = %d: \t==========================\n", threadIdx.x);
		///printf("\tthreadIdx.x = %d: \tfind_solution_borders_cpu:\n", threadIdx.x);
		///printf("\tthreadIdx.x = %d: \t--------------------------\n", threadIdx.x);
		///printf("\tthreadIdx.x = %d: \t[%f, %f] - borders\n", threadIdx.x, border_left, border_right);
		///printf("\tthreadIdx.x = %d: \twidth = %f\n", threadIdx.x, width);
		///printf("\tthreadIdx.x = %d: \tintervals_n = %d\n", threadIdx.x, intervals_n);
		///printf("\tthreadIdx.x = %d: \tFOUND solution in the %d thread!\n", threadIdx.x, threadIdx.x);

		borders_ptr_dev[0] = x_;
		borders_ptr_dev[1] = x;
		///printf("\tthreadIdx.x = %d: \tf(x) in [%f, %f]\n", threadIdx.x, f_x_, f_x);
		///printf("\tthreadIdx.x = %d: \tx    in [%f, %f]\n", threadIdx.x, borders_ptr_dev[0], borders_ptr_dev[1]);
		///printf("\tthreadIdx.x = %d: \t==========================\n\n", threadIdx.x);
	}
	__syncthreads();

	// FIXME (get rid of gap cheking by making search a bit wider)
	// Checking te GAP
	/*
	printf("\tChecking if there's solution beeween [x, right_border]\n");

	x = border_right;  //for gpu
	f_x = calculate_function_gpu(a, b, x);

	if ((f_x_ > 0) && (f_x < 0) || (f_x_ < 0) && (f_x > 0)) {
		printf("\tSolution IS FOUND in the LAST GAP!\n");
	}
	//*/


	//////////////

	//return borders_ptr;
}



__host__ float find_solution_gpu_on_host(float a, float b,
		float border_left, float border_right,
		float pos_x, float eps, int intervals_n) {
	printf("=======================\n");
	printf("find_solution_gpu_on_host\n");
	printf("-----------------------\n");
	
	/*
	int solutiouns_n = 1;	//AMOUNT OF ABLE SOLUTIONS
	float* solutiouns = new float[solutiouns_n];
	float* solutiouns_dev = NULL;
	cudaMalloc((void**)&solutiouns_dev, solutiouns_n * sizeof(float));
	//dim3 threads = dim3(32, 1, 1);
	//dim3 blocks  = dim3(intervals_n / threads.x, 1, 1);
	float* solutiouns_dev = (float*)malloc(1 * sizeof(float));
	//*/

	int borders_n = 2 * 1; // borders (left and right) amount	-	2 * N 
	// Dynamically allocate memory using malloc() on CPU
	float* borders_ptr = (float*)malloc(borders_n * sizeof(float));
	// Dynamically allocate memory using malloc() on GPU
	float* borders_ptr_dev = NULL;

	// init before cycle
	borders_ptr[0] = border_left;
	borders_ptr[1] = border_right;
	float radius = borders_ptr[1] - borders_ptr[0];
	float width = radius / intervals_n;
	eps *= 2;
	printf("[%f, %f] - borders\n", borders_ptr[0], borders_ptr[1]);
	printf("radius = %f, \t", radius);
	printf("width  = %f\n", width);
	printf("eps = %f\n", eps);

	int iterations = 0;
	while (radius > eps) {
		++iterations;
		///printf("000000.\n");
		cudaMalloc((void**)&borders_ptr_dev, borders_n * sizeof(float));
		find_solution_borders_gpu_global<<<1, intervals_n>>> (borders_ptr_dev,
				a, b, borders_ptr[0], borders_ptr[1], width, pos_x);
		//__syncthreads();
		///printf("111111.\n");
		//cudaDeviceSynchronize();
		cudaMemcpy(borders_ptr, borders_ptr_dev, borders_n * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceReset();
		radius = borders_ptr[1] - borders_ptr[0]; // abs() suppports only integer values!
		width = radius / intervals_n;
		printf("RESULT %d: x in [%f, %f]\n", iterations, borders_ptr[0], borders_ptr[1]);
		///printf("radius = %f\t", radius);
		///printf("width  = %f\n\n", width);

		///printf("222222.\n");
		//system("pause");
	}
	printf("=======================\n");
	printf("FINAL RESULT: x in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	float result = borders_ptr[0] + (borders_ptr[1] - borders_ptr[0]) / 2;
	eps /= 2;

	printf("x = %f +- %f\n", result, eps);
	printf("Iterations = %d\n", iterations);
	printf("radius = %f\n", radius);
	printf("eps    = %f\n", eps);
	printf("=======================\n\n");
	cudaFree(borders_ptr_dev);
	free(borders_ptr);
	return result;
}



double count_solution_time_gpu_host(float a, float b,
		float border_left, float border_right,
		float pos_x, float eps, int intervals_n) {
	clock_t begin = clock();
	///float result = 
	find_solution_gpu_on_host(a, b,
		border_left, border_right, 
		pos_x, eps, intervals_n
		);

	///printf("Result = %f +- %f\n", result, eps);
	//cudaDeviceReset();
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


int func(void (*f)(int)) {
	return 1;
};

void do_smth(int a) { printf("%d\n", a); };

int main(void) {
	//read equasion
	//func(do_smth);
	//printf("%d\n");
	//parse equasion


	//rewrite equasion in needed format: f(x) = 0;


	//get_cpu_calculations_time


	//calculate_using_gpu_calculations_time


	//printf("\nPress 'Enter' to write numbers using CPU..\n");
	//getchar();


	// float pos_x = 0; // ЛУЧШЕ ВСЕГДА ТАК, чтобы не обрабатывать лишнюю ситуацию (когда решение лежит в [border_left, border_left + pos_x])
	// FIXME: в данном случае нам при условии (x_ < border_right) & (x > border_right) нужно проверить есть ли решение в [x, border_right]
	float a = 2., b = 3.,
		border_left = -10000., 
		border_right = 10000.,
		pos_x = 0, // ЛУЧШЕ ВСЕГДА ТАК, чтобы не обрабатывать лишнюю ситуацию
		eps = 0.1;
	int intervals_n = N;

	double cpu_time = count_solution_time_cpu(a, b, border_left, border_right, pos_x, eps, intervals_n);
	printf("\nPress 'Enter' to write numbers using GPU..\n");
	//system("pause");
	//getchar();
	double gpu_time = count_solution_time_gpu_host(a, b, border_left, border_right, pos_x, eps, intervals_n);
	printf("\n\n----------------------------------------------------\n");
	printf("Parameters:\n\tN = %d\n", N);
	printf("\n-------------\n");
	printf("Results:\n");
	printf("\tFinding the solution on CPU took %.2f seconds\n", cpu_time);
	printf("\tFinding the solution on GPU took %.2f seconds\n", gpu_time);
	printf("----------------------------------------------------\n\n");
	printf("Wow!!\n");
}




/*Типы памяти в CUDA*/
/*
см. презентацию
Shared быстрая память видит вся видеокарта
constant
texture


|------------------------------------------------------|
| Спецификатор	|	  Доступ	|	  Вид доступа	   |
|  переменных	|				|					   |
|-------------------------------|----------------------|
| __device__	|	  device	|	 R				   |
| __constant__	|  host/device	|  R / W			   |
| __shared__	|	  block		| RW / __syncthreads() |
|------------------------------------------------------|
P.s. Все они находятся на устройстве.


В текстурной памяти есть кэш
доступ к ней идёт через: tex1D, tex2D, tex1Dfetch

tex1Dfetch(tex, int)	  -	линейная (только для однмерных массивов,)
tex1D(), tex2D(), tex3D() -	texcudaArray


Существуют различные виды хранения данных, например можно отобразить нормализованные значения в

gridDim
blockDim

*/