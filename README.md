# Parallel and Distributed Computing (PaDC)
Doing my labs here...

## Lab 1 - Getting started...
---
The task is simple-dimple as... that:
### __Task__ 
Print out numbers form 1 to 65535 using CUDA.

### __Solution__ 

We create function *__print_gpu__* on the device (GPU) which would be called from the CPU.
```
__global__ void print_gpu(char separator) {
	printf("%d%c", blockIdx.x + 1, separator); 
}
```
That fuction should be called with the next line of code (in our case n=65535):
```
print_gpu<<<n, 1>>>(separator); // n blocks, 1 thread
```


### __Conclusions__ (but cmon... what could they be...)
Yea, GPU is REALLY faster. Especiaclly when we are passing newline character '\n' as a parameter to print_gpu.


## Lab 2 - Using different types of memory
---


### __Task__
We need to find the solutions of any equasion. 

For the simplicity I've chosen the following linear equasion:

__a?x + b = 0__.

### __Solution__ 

We bring the equation to the following form: **f(x) = 0**.  After that we:

1. determine the boundaries on which the solution is located.
2. split this area into equal intervals
3. evaluate the value of ***f(x)*** on each segment and
4. look for 2 adjacent segments where the values of ***f(x)*** have opposite signs.

If such intervals were not found, that may be caused by next reasons:

- *There are no solutions within the given boundaries*
- *__(Not in our case)__ The segment is split too wide.* Because of this, we can skip the place where the function passes through zero, since it manages to go back. Thus, we can lose not one, but two solutions at once.

#### Variants of Implementation 

##### Implementation 1 (using ```__global__``` memory)
**Pros (+):** The loop is on the CPU so as not to take up a bunch of GPU threads for no good reason. Otherwise, it would perform the same action with the same parameters in all threads allocated to it.

**Cons (-):** Unfortunately, in this case only slow *global memory* can be used.

```
__global__  find_borders() {
	...
}

__host__ find_solution(...) {
	while (radius > epsilon) {
		...
	}
}

__host__ int main() {
    ...
    find_solution(...);
    ...
}
```

##### Implementation 2 (using ```__shared__``` memory)
**Pros (+):**  The faster (shared) memory that is inside the block can be used.

**Cons (-):** The loop is on the GPU, which is why a lot of threads are engaged in it just for the same action with the same parameters.
```
__device__  find_borders() {
	...
}

__global__ find_solution(...) {
	while (radius > epsilon) {
		...
	}
}

__host__ int main() {
    ...
    find_solution<<<...>>>(...);
    ...
}
```

### __Conclusions__
When comparing the implementation of the task on the CPU and the GPU, the performance gains quite strongly leaned towards the CPU. The gap between GPU and CPU is significant for any N. This is so due to the ***peculiarity of the algorithm: it uses a loop***. Since the clock speed of the processor is higher than the clock speed of the video card, the cycle is faster in it. The advantages of the GPU are in performing *the same operations with __different__ parameters*. In our case (in the implementation through shared memory), it has to perform *the same operations with __the same__ parameters* in a loop. More explicitly, the conclusion that the chosen algorithm is not optimal for a GPU can be confirmed by the fact that the implementation through a slower global memory has shown itself to be better, since the loop is executed on the CPU. Performance analysis was carried out on the basis of time indicators calculated directly in the program code.

The difference between the execution time of a GPU (with global memory and a cycle on the CPU) and a CPU with global memory and a CPU has been decreasing rather slowly. From this we can conclude that __it is better to use CPU__ to solve problems that require a loop with the same parameters.

## Lab 3
---
### __Task__
### __Solution__ 
### __Conclusions__