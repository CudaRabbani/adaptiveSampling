/*
 * reconstruction.h
 *
 *  Created on: Feb 1, 2017
 *      Author: reza
 */

#ifndef RECONSTRUCTION_H_
#define RECONSTRUCTION_H_

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h"


void initializeConvolutionFilter(float *kernel, int kernelLength);
void reconstructionFunction(dim3 grid, dim3 block, float *red, float *green, float *blue, int *pattern, float *red_res, float *green_res, float *blue_res,
 		int dataH, int dataW, float *device_x, float *device_p);
/*
void reconstructionFunction(dim3 grid, dim3 block, float *data, float *red, float *green, float *blue,
		int *pattern, float *kernel, float *d_result,float *red_res, float *green_res, float *blue_res,
		int maskH, int maskW, int dataH, int dataW, float *device_x, float *device_p);
*/
void VectorDotProduct(dim3 gridSize, dim3 blockSize, float *data_a, float *data_b, float *d_result, int length, int width);
__global__ void reconstructionKernel(float *data, float *result, int *pattern, int dataH, int dataW, volatile float *device_x, volatile float *device_p);
//__global__ void reconstructionKernel(float *data, float *result, int *pattern, int dataH, int dataW, volatile float *device_x, volatile float *device_p)

void testAdd(dim3 grid, dim3 block, float *d_a, int size, float *ans);
#endif /* RECONSTRUCTION_H_ */
