#include "deviceVars.h"
#include "reconstruction.h"
#include "constantMemory.h"


#define MASK_W 7
#define MASK_H 7
#define TILE_W 16 //It has to be same size as block
#define TILE_H 16 //It has to be same size as block
#define MASK_R (MASK_W / 2)

#define w (TILE_W + MASK_W -1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define ThreadPerBlock TILE_H*TILE_W
#define PAD 3

cudaStream_t R, G, B;
cudaEvent_t Recon_start, Recon_end;
float recon_time;

__device__ void StSPk_Operation(float *d_Vector, float *d_x, int *pattern);
__device__ void additionScalar(float *d_Vector, float *d_first, float *d_second, float scalar);
__device__ void multiplyA(float *d_Vector, float *device_x, float *d_x, int *pattern, float *convResult, int dataH, int dataW, float *temp, float *holoArray);
__device__ void dotProduct(volatile float *cache, float *temp);
__device__ void convolve(float *data, float *temp, float *convResult, float *holoArray, int dataH, int dataW);




__device__ __constant__ float MASK[MASK_W * MASK_H];
__constant__ int_2 d_temp[7][256];

__global__ void reconstructionKernel(float *data, float *result, int *pattern, int *d_varPriority, int dataH, int dataW, volatile float *device_x, volatile float *device_p)
{

    __shared__ float holoArray[w*w]; //contains holo elements
    __shared__ float holoArrayNull[w*w];
//    __shared__ float holoTemp[w*w]; //contains temporary data for convolution
	__shared__ float temp[ThreadPerBlock];
    __shared__ float convResult[ThreadPerBlock];
    __shared__ float d_Vector[ThreadPerBlock];
    __shared__ float d_current_x[ThreadPerBlock];
    __shared__ float d_current_r[ThreadPerBlock];
    __shared__ float d_current_p[ThreadPerBlock];
    __shared__ float d_next_x[ThreadPerBlock];
    __shared__ float d_next_r[ThreadPerBlock];
    __shared__ float d_next_p[ThreadPerBlock];
    volatile __shared__ float cache_crnt_r[ThreadPerBlock]; //for dot product only
    volatile __shared__ float cache_crnt_p[ThreadPerBlock]; //for dot product only
    volatile __shared__ float cache_next_r[ThreadPerBlock]; //for dot product only
    __shared__ float cache[ThreadPerBlock];
    __shared__ int pixels[ThreadPerBlock];
    int local_x, local_y;


    float dot_Num;
    float dot_Denom;
    float dot_alpha;
    float dot_beta;

    int GW = gridDim.x * blockDim.x + (gridDim.x + 1) * PAD;
    int GH = gridDim.y * blockDim.y + (gridDim.y + 1) * PAD;
    int STRIPSIZE = GW * (blockDim.y + PAD);


    float flag = 1.0f;
    int counter=0;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * dataW;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tx = x + (blockIdx.x +1) * PAD;
	int ty = y + (blockIdx.y +1)* PAD;

	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;

	int row = d_varPriority[bid];
//	if(localIndex == 0)
//	{
//		pixels[localIndex] = 0;
//	}
//	__syncthreads();
/*
	if(localIndex == 0)
	{
		for(int i=0; i< ThreadPerBlock; i++)
		{
			local_x = d_temp[row][i].x;
			local_y = d_temp[row][i].y;
			int tempLin = local_x + local_y * 16;
			if(local_x != 999 && local_y != 999)
			{

//				pattern[i] = 1;
				pixels[tempLin] = 1;
			}

		}
	}

	__syncthreads();
*/
	if(localIndex == 0)
	{
		int holoCounter = 0;
		int corner_x = blockIdx.x * blockDim.x + (blockIdx.x+1)* PAD;
		int corner_y = blockIdx.y * blockDim.y + (blockIdx.y+1)* PAD;
		for(int j= (corner_y - PAD); j<(corner_y+PAD+blockDim.y); j++)
		{
			for(int i = (corner_x - PAD); i<(corner_x+PAD+blockDim.x); i++)
			{
				int imageId = i + dataW * j;
				holoArray[holoCounter] = data[imageId];
				holoCounter++;
			}
		}
	}

	__syncthreads();
	if(localIndex == 1)
	{
		int holoCounter = 0;
		int corner_x = blockIdx.x * blockDim.x + (blockIdx.x+1)* PAD;
		int corner_y = blockIdx.y * blockDim.y + (blockIdx.y+1)* PAD;
		for(int j= (corner_y - PAD); j<(corner_y+PAD+blockDim.y); j++)
		{
			for(int i = (corner_x - PAD); i<(corner_x+PAD+blockDim.x); i++)
			{
//				int imageId = i + dataW * j;
				holoArrayNull[holoCounter] = 0.0f;
				holoCounter++;
			}
		}
	}
	__syncthreads();


	d_current_x[localIndex] = device_x[haloIndex];//data[index];
	cache[localIndex] = data[haloIndex];
	pixels[localIndex] = pattern[haloIndex];
	__syncthreads();
	multiplyA(d_Vector, d_current_x, d_current_x, pixels, convResult, dataH, dataW,temp, holoArray);

	__syncthreads();

	additionScalar(d_current_r, cache, d_Vector, -1); //cache = d_b; r = b - Ax
	d_current_p[localIndex] = d_current_r[localIndex];
	device_p[haloIndex] = d_current_p[localIndex];


	__syncthreads();

	// (fabs(flag - 0.00) > 1e-2) (fabs(flag - 0.00) > 1e-6) && (counter < 3) && (counter < 50)    fabs(flag - 0.00) > 1e-6


	while (fabs(flag - 0.00) > 1e-6) //fabs(flag - 0.00) > 1e-6			counter < 50
		{
			//Dot product goes here and the answer will be stored in dot_result_num
			cache_crnt_r[localIndex] = d_current_r[localIndex]*d_current_r[localIndex];
			__syncthreads();

			dotProduct(cache_crnt_r, &dot_Num);
			__syncthreads();

			multiplyA(d_Vector, d_current_p, d_current_p, pixels,convResult,dataH, dataW, temp, holoArrayNull);
			__syncthreads();

			cache_crnt_p[localIndex] = d_current_p[localIndex] * d_Vector[localIndex];
			__syncthreads();
			dotProduct(cache_crnt_p, &dot_Denom);
			__syncthreads();
			dot_alpha = dot_Num / dot_Denom;
			additionScalar(d_next_x, d_current_x, d_current_p, dot_alpha);
			additionScalar(d_next_r, d_current_r,d_Vector, (-1)* dot_alpha);
			cache_next_r[localIndex] = d_next_r[localIndex] * d_next_r[localIndex];
			__syncthreads();
			dotProduct(cache_next_r, &dot_Denom); //beta = next_r/current_r
			__syncthreads();
			flag = sqrtf(dot_Denom);
			dot_beta = dot_Denom / dot_Num;
			additionScalar(d_next_p, d_next_r,d_current_p, dot_beta);
			d_current_r[localIndex] = d_next_r[localIndex];
			d_current_p[localIndex] = d_next_p[localIndex];
			d_current_x[localIndex] = d_next_x[localIndex];
			counter++;
			__syncthreads();
		}

	result[haloIndex] = d_next_x[localIndex];
//	pattern[haloIndex] = pixels[localIndex];



}

__device__ void convolve(float *data, float *temp, float *convResult, float *holoArray, int dataH, int dataW)
{
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;


    temp[localIndex] = data[localIndex];
    __syncthreads();

    float out = 0.0f;

    int corner_x = threadIdx.x - MASK_W/2;
    int corner_y = threadIdx.y - MASK_H/2;
    for(int y = 0; y<MASK_H; y++)
    {
   	 for(int x = 0; x<MASK_W; x++)
   	 {
   		 int i = corner_x + x;
   		 int j = corner_y + y;
   		 int maskIndex = x + y * MASK_W;
   		 int imageIndex;
   		 int holoIndex;
   		 float imageData;
   		 if(i<0 || i>=blockDim.x || j<0 || j>=blockDim.y)
   		 {
   			 i+=MASK_W/2;
   			 j+=MASK_H/2;
   			 holoIndex = i + j * (blockDim.x + 2 * PAD);
   			 imageData = holoArray[holoIndex];
   		 }
   		 else
   		 {
   			 imageIndex = i + j * blockDim.x;
   			 imageData = temp[imageIndex];
   		 }
   		 out += MASK[maskIndex] * imageData;
   	 }

    }
    convResult[localIndex] = out; //writing convolution result in shared memory for that block;
}

__device__ void multiplyA(float *d_Vector, float *device_x, float *d_x, int *pattern, float *convResult, int dataH, int dataW, float *temp, float *holoArray)
{
    convolve(device_x, temp, convResult, holoArray, dataH, dataW); //result will be also written on shared memory convResult;
    StSPk_Operation(d_Vector, d_x, pattern); //result will be also stored on temp shared memory
    additionScalar(d_Vector,d_Vector,convResult,1.0f); //result will be stored in result
}

__device__ void StSPk_Operation(float *d_Vector, float *d_x, int *pattern)
{
    int localIndex = threadIdx.x + threadIdx.y * TILE_W;
    d_Vector[localIndex] = d_x[localIndex] * pattern[localIndex];
}

__device__ void additionScalar(float *d_Vector, float *d_first, float *d_second, float scalar)
{
    int localIndex = threadIdx.x + threadIdx.y * TILE_W;
    d_Vector[localIndex] = d_first[localIndex] + scalar*d_second[localIndex];
}

 __device__ void dotProduct(volatile float *cache, float *temp)
{

    int localIndex = threadIdx.x + threadIdx.y * blockDim.x;

    if( localIndex < 128) {
    	cache[localIndex] += cache[localIndex + 128];
    }
    __syncthreads();
    if( localIndex < 64) {
        	cache[localIndex] += cache[localIndex + 64];
    }
    __syncthreads();
    if( localIndex < 32) {
    	cache[localIndex]+=cache[localIndex+32];
    	cache[localIndex]+=cache[localIndex+16];
    	cache[localIndex]+=cache[localIndex+8];
    	cache[localIndex]+=cache[localIndex+4];
    	cache[localIndex]+=cache[localIndex+2];
    	cache[localIndex]+=cache[localIndex+1];
    }

    __syncthreads();
    temp[0] = cache[0];

}

void initializeConvolutionFilter(float *kernel, int kernelLength)
{
	if(cudaMemcpyToSymbol(MASK, kernel, kernelLength * sizeof(float)) != cudaSuccess)
	 {
		 printf("Copy to constant memory error\n");
	 }
	else printf("copy to MASK successful\n");


}
 void reconstructionFunction(dim3 grid, dim3 block, float *red, float *green, float *blue,
 		int *pattern, int *d_varPriority, float *res_red, float *res_green, float *res_blue, int dataH, int dataW, float *device_x, float *device_p)
 {
/*
	 cudaEventCreate(&Recon_start);
	 cudaEventRecord(Recon_start, 0);
	 */
	cudaStreamCreate(&R);
	cudaStreamCreate(&G);
	cudaStreamCreate(&B);

     reconstructionKernel<<<grid,block,0,R>>>(red, res_red, pattern, d_varPriority, dataH, dataW, device_x, device_p);
     reconstructionKernel<<<grid,block,0,G>>>(green, res_green, pattern, d_varPriority, dataH, dataW, device_x, device_p);
     reconstructionKernel<<<grid,block,0,B>>>(blue, res_blue, pattern, d_varPriority, dataH, dataW, device_x, device_p);

     cudaStreamDestroy(R);
     cudaStreamDestroy(G);
     cudaStreamDestroy(B);
/*
	 reconstructionKernel<<<grid,block>>>(red, res_red, pattern, dataH, dataW, device_x, device_p);
	      reconstructionKernel<<<grid,block>>>(green, res_green, pattern, dataH, dataW, device_x, device_p);
	      reconstructionKernel<<<grid,block>>>(blue, res_blue, pattern, dataH, dataW, device_x, device_p);
	      */
     cudaDeviceSynchronize();

     getLastCudaError("kernel failed for reconstruction\n");


 }

 __global__ void testAddKernel(float *d_a, int size, float *ans)
 {

	 __shared__ float temp[TILE_W*TILE_H];
	 int localIndex = threadIdx.x;// + threadIdx.y * blockDim.x;
	 temp[localIndex] = d_a[localIndex];
	 __syncthreads();

	 dotProduct(temp, ans);

 }


 void testAdd(dim3 grid, dim3 block, float *d_a, int size, float *ans)
 {
	 testAddKernel<<<grid,block>>>(d_a, size, ans);
 }

__global__ void testConstant()
{
	int x = 5;
	int y = 6;
 	printf("Reconstruction: %d, %d\n", d_temp[x][y].x, d_temp[x][y].y);
}

 void copyConstantTestReconstruction(dim3 grid, dim3 block, int_2 temp[7][256])
 {
 	checkCudaErrors(cudaMemcpyToSymbol(d_temp, temp, 7*256*sizeof(int_2)));
 	testConstant<<<1,1>>>();

 }
