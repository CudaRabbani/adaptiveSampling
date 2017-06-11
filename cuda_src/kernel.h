/*
 * kernel.h
 *
 *  Created on: Jan 30, 2017
 *      Author: reza
 */

#ifndef KERNEL_H_
#define KERNEL_H_




typedef unsigned short ushort;
typedef struct
{
	int x;
	int y;
}int_2;

void setTextureFilterMode(bool bLinearFilter);
void initCuda(void *h_volume, cudaExtent volumeSize);
void freeCudaBuffers();
void render_kernel(dim3 gridVol, dim3 gridVolStripe, dim3 blockSize, float *d_var, int *d_varPriority, int *d_pattern, int *d_linear, int *d_xPattern, int *d_yPattern, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue,
		float *res_red, float *res_green, float *res_blue, float *device_x, float *device_p, int imageW, int imageH, float density, float brightness, float transferOffset,
		float transferScale,bool isoSurface, float isoValue, bool lightingCondition, float tstep, bool cubic, bool cubicLight, int filterMethod, int *d_linPattern, int *d_X, int *d_Y, int onPixel, int stripePixels);
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void blendFunction(dim3 grid, dim3 block,int *d_varPriority, bool reconstruct, int *d_linPattern, uint *d_output, float *d_res, float *d_green, float *d_blue, float *res_red, float *res_green, float *res_blue, int imageH, int imageW);
//void initPixelBuffer();
//    blendFunction(gridVol, blockSize, d_output,d_vol, res_red, res_green, res_blue, height, width, d_xPattern, d_yPattern, d_linear);
int iDivUp(int a, int b);
void initCudaCubicSurface(const uchar* voxels, uint3 volumeSize);
//void initCudaCubicSurface(const ushort* voxels, uint3 volumeSize);
void varianceFunction(dim3 grid, dim3 block, float *input, float *output, int dataH, int dataW);
void copyTenPercentage(int *pixels);
void copyAllPercentageRenderer(int *ten, int *twenty, int *thirty, int *fourty, int *fifty, int *sixty, int *seventy, int *eighty);

void generateAddress(dim3 grid, dim3 block, int *variance, int *X, int *Y, int *linear);

void copyConstantTest(dim3 grid, dim3 block, int temp[7][256]);
void copyConstantTest_1(dim3 grid, dim3 block, int_2 temp[7][256]);


#endif /* KERNEL_H_ */
