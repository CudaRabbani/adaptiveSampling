/*
 * kernel.h
 *
 *  Created on: Jan 30, 2017
 *      Author: reza
 */

#ifndef KERNEL_H_
#define KERNEL_H_




typedef unsigned short ushort;


void setTextureFilterMode(bool bLinearFilter);
void initCuda(void *h_volume, cudaExtent volumeSize);
void freeCudaBuffers();
void render_kernel(dim3 gridSize, dim3 blockSize, int *d_pattern, int *linPattern, int *d_xPattern, int *d_yPattern, float *d_vol, float *d_red, float *d_green, float *d_blue,
float *res_red, float *res_green, float *res_blue, float *device_x, float *device_p, int imageW, int imageH, float density, float brightness, float transferOffset,
float transferScale,bool isoSurface, float isoValue, bool lightingCondition, float tstep, bool cubic, bool cubicLight, int filterMethod, float *d_temp);
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void blendFunction(dim3 grid, dim3 block,uint *d_output, float *d_vol, float *res_red, float *res_green, float *res_blue, int imageH, int imageW, int *d_xPattern, int *d_yPattern, int *d_linear);
//void initPixelBuffer();
//    blendFunction(gridVol, blockSize, d_output,d_vol, res_red, res_green, res_blue, height, width, d_xPattern, d_yPattern, d_linear);

void initCudaCubicSurface(const uchar* voxels, uint3 volumeSize);
//void initCudaCubicSurface(const ushort* voxels, uint3 volumeSize);
void varianceFunction(dim3 grid, dim3 block, float *input, float *output, int dataH, int dataW);
void copyTenPercentage(int *pixels);






#endif /* KERNEL_H_ */
