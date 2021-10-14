//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// if N >= 2147483647 then error
#define N 322
#define SEPARATOR ','

//using namespace std;


// returns array of solutions
/*
double* find_equasion_solutions_gpu(function, border_left, width, pos_x) {
	// Определить какой промежуток мы используем
	// (и точку в которой считаем значение функции)
	double thread_x = border_left + (blockIdx.x + pos_x) * width;

	// count function value
	// (may be bettered by analysing the equasion or by using less float values)


	// compare resuslt with 0

}
//*/

// returns borders
//double* find_solution_borders_cpu(double* borders_ptr, double a, double b, double border_left, double border_right, int borders_n, double pos_x) {
double* find_solution_borders_cpu(double* borders_ptr, double a, double b, double border_left, double border_right, double width, double pos_x) {

	printf("border_right = %f\n", border_right);
	printf("border_left = %f\n", border_left);
	printf("width = %f\n", width);
	//printf("border_right - border_left = %f\n", border_right - border_left);
	//printf("(border_right - border_left) / width = %f\n", (border_right - border_left) / width);
	int borders_n = ((border_right - border_left) / width);
	printf("borders_n = %d\n", borders_n);
	system("pause");
	if (borders_n < 2) {
		printf("ERROR: borders_n = %d\n", borders_n);
		system("pause");
		exit(1);
	}
	if ((border_right - border_left) < width * 2) {
		printf("ERROR: incorrect width;  borders_n = %d\n", borders_n);
		system("pause");
		exit(1);
	}

	double f_x, f_x_, x, x_;

	// Определить какой промежуток мы используем 
	// (и точку в которой считаем значение функции для каждой нити)
	int idx = 0;
	for(; idx < borders_n + 1; idx++) {
		// double x = border_left + (blockIdx.x + pos_x) * width;  //for gpu
		x = border_left + (idx + pos_x) * width;  // for cpu

		// count function value 
		// (may be bettered by analysing the equasion or by using less float values)
		f_x = a + b * x;
		if (idx == 0) {
			f_x_ = f_x;
		}


		// compare resuslt with 0
		if ((f_x_ > 0) & (f_x < 0) || (f_x_ < 0) & (f_x > 0)) {
			break;
		}
		//remember if prev was less or more than 0

		x_ = x;
		f_x_ = f_x;
		idx++;
	}

	//(x_ < border_right) & (x > border_right)

	borders_ptr[0] = x_;
	borders_ptr[1] = x;

	printf("x    in [%f, %f]\n", borders_ptr[0], borders_ptr[1]);
	printf("f(x) in [%f, %f]\n", f_x_, f_x);
	//getchar();

	return borders_ptr;
}


double count_solution_time_cpu(double a, double b, double border_left, double border_right, double width, double pos_x, double eps, int n) {
	clock_t begin = clock();

	// Dynamically allocate memory using malloc()
	double* borders_ptr = (double*)malloc(2 * sizeof(double));
	if (borders_ptr == NULL) {
		printf("ooooooops\n");
		getchar();
		exit(1);
	}
	borders_ptr[0] = border_left;
	borders_ptr[1] = border_right;


	double radius = abs(border_right - border_left);

	printf("radius = %f,\t border_left = %f,\t border_right = %f]\n", radius, borders_ptr[0], borders_ptr[1]);
	printf("eps=%f\n", eps);

	while (radius > eps) {
		// double* find_solution_borders_cpu(double* borders_ptr, double a, double b, double border_left, int borders_n, double border_right, double pos_x)
		borders_ptr = find_solution_borders_cpu(borders_ptr, a, b, borders_ptr[0], borders_ptr[1], width, pos_x);
		printf("RESULT : x in[%f, %f]\n", borders_ptr[0], borders_ptr[1]);
		radius = borders_ptr[1] - borders_ptr[0];  // abs() suppports only integer values!
		printf("radius    = %f\n\n", radius);
		width = radius / n;
		//printf("width = %f\nradius = %f\n[border_left = %f, border_right = %f]\n", width, radius, borders_ptr[0], borders_ptr[1]);
		//system("pause");
	}
	printf("FINAL RESULT : x in [%f, %f]\n\n", borders_ptr[0], borders_ptr[1]);
	printf("radius = %f\n", radius);
	printf("eps    = %f\n", eps);

	free(borders_ptr);

	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

/*
double count_solution_time_gpu(int n) {
	clock_t begin = clock();
		print_gpu << <n, 1 >> > (separator); // n блоков, 1 нить в каждом
		cudaDeviceReset();
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}
//*/


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

	double pos_x = 0; // ЛУЧШЕ ВСЕГДА ТАК, чтобы не обрабатывать лишнюю ситуацию (уогда решение лежит в [border_left, border_left + pos_x])
	// FIXME: в данном случае нам при условии (x_ < border_right) & (x > border_right) нужно посчитать значение в border_right


	//printf("\nPress 'Enter' to write numbers using CPU..\n");
	//getchar();
	double cpu_printing_time = count_solution_time_cpu(2., 3., -10., 10., 2., pos_x, 0.01, 2);
	printf("\nPress 'Enter' to write numbers using GPU..\n");
	//getchar();
	//double gpu_printing_time = count_solution_time_gpu(N, SEPARATOR);
	printf("\n\n----------------------------------------------------\n");
	printf("Parameters:\n\tN = %d\n\tSEPARATOR = '%c'", N, SEPARATOR);
	printf("\n-------------\n");
	printf("Results:\n");
	printf("\tPrinting time using CPU took %.2f seconds\n", cpu_printing_time);
	//printf("\tPrinting time using GPU took %.2f seconds\n", gpu_printing_time);
	printf("----------------------------------------------------\n\n");
	printf("Wow!!\n");
}




/*Типы памяти в CUDA*/
/*
см. презентацию
Shared быстрая память видит вся видеокарта
constant 
texture 


__device__
__constant__
__shared__ 



В текстурной памяти есть кэш
доступ к ней идёт через: tex1D, tex2D, tex1Dfetch

tex1Dfetch(tex, int)	  -	линейная (только для однмерных массивов,)
tex1D(), tex2D(), tex3D() -	texcudaArray


Существуют различные виды хранения данных, например можно отобразить нормализованные значения в

gridDim
blockDim

*/