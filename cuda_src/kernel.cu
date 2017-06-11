#include "helper_math.h"
#include "helper_functions.h"
#include "CI/memcpy.cu"
#include "CI/cubicPrefilter3D.cu"
#include "CI/cubicTex3D.cu"

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include "deviceVars.h"
#include "reconstruction.h"

#define TILE_W 16 //It has to be same size as block
#define TILE_H 16 //It has to be same size as block
#define PAD 3



cudaStream_t stripe, blocks;
cudaStream_t grOne, grTwo, grThree;
cudaEvent_t start, stop;
float volumeTime;

typedef unsigned int  uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

cudaArray *d_volumeArray = 0;
cudaArray *volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture
texture<float4, 1, cudaReadModeElementType>         transferTexIso;
texture<uchar, 3, cudaReadModeNormalizedFloat> tex_cubic;
//texture<ushort, 3, cudaReadModeNormalizedFloat> tex_cubic;
texture<float, 3, cudaReadModeElementType> coeffs;



typedef struct
{
	float4 m[3];
} float3x4;
typedef struct
{
	int x;
	int y;
}int_2;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

__constant__ int TenPercent[TILE_W*TILE_H];
__constant__ int TwentyPercent[TILE_W*TILE_H];
__constant__ int ThirtyPercent[TILE_W*TILE_H];
__constant__ int FourtyPercent[TILE_W*TILE_H];
__constant__ int FiftyPercent[TILE_W*TILE_H];
__constant__ int SixtyPercent[TILE_W*TILE_H];
__constant__ int SeventyPercent[TILE_W*TILE_H];
__constant__ int EightyPercent[TILE_W*TILE_H];

__constant__ int_2 d_temp[7][256];



struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__ float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__ float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void setTextureFilterMode(bool bLinearFilter)
{
	tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

void initCuda(void *h_volume, cudaExtent volumeSize)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	//	cudaPitchedPtr d_volumeMem;
	//	size_t size = d_volumeMem.pitch * volumeSize.height * volumeSize.depth;
	//	h_volume = (VolumeType*)malloc(size);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeBorder;  // clamp texture coordinates //cudaAddressModeClamp //cudaAddressModeBorder
	tex.addressMode[1] = cudaAddressModeBorder;
	tex.addressMode[2] = cudaAddressModeBorder;
	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
	/*
    // create transfer function texture
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };
	 */
	float4 transferFunc[] =
	{
			{0.231372549,	0.298039216,	0.752941176,	0,},
			{0.266666667,	0.352941176,	0.8,	0.03125,},
			{0.301960784,	0.407843137,	0.843137255,	0.0625,},
			{0.341176471,	0.458823529,	0.882352941,	0.09375,},
			{0.384313725,	0.509803922,	0.917647059,	0.125,},
			{0.423529412,	0.556862745,	0.945098039,	0.15625,},
			{0.466666667,	0.603921569,	0.968627451,	0.1875,},
			{0.509803922,	0.647058824,	0.984313725,	0.21875,},
			{0.552941176,	0.690196078,	0.996078431,	0.25,},
			{0.596078431,	0.725490196,	1,	0.28125,},
			{0.639215686,	0.760784314,	1,	0.3125,},
			{0.682352941,	0.788235294,	0.992156863,	0.34375,},
			{0.721568627,	0.815686275,	0.976470588,	0.375,},
			{0.760784314,	0.835294118,	0.956862745,	0.40625,},
			{0.800000000,	0.850980392,	0.933333333,	0.4375,},
			{0.835294118,	0.858823529,	0.901960784,	0.46875,},
			{0.866666667,	0.866666667,	0.866666667,	0.5,},
			{0.898039216,	0.847058824,	0.819607843,	0.53125,},
			{0.925490196,	0.827450980,	0.772549020,	0.5625,},
			{0.945098039,	0.8,	0.725490196,	0.59375,},
			{0.960784314,	0.768627451,	0.678431373,	0.625,},
			{0.968627451,	0.733333333,	0.62745098,	0.65625,},
			{0.968627451,	0.694117647,	0.580392157,	0.6875,},
			{0.968627451,	0.650980392,	0.529411765,	0.71875,},
			{0.956862745,	0.603921569,	0.482352941,	0.75,},
			{0.945098039,	0.552941176,	0.435294118,	0.78125,},
			{0.925490196,	0.498039216,	0.388235294,	0.8125,},
			{0.898039216,	0.439215686,	0.345098039,	0.84375,},
			{0.870588235,	0.376470588,	0.301960784,	0.875,},
			{0.835294118,	0.31372549,	0.258823529,	0.90625,},
			{0.796078431,	0.243137255,	0.219607843,	0.9375,},
			{0.752941176,	0.156862745,	0.184313725,	0.96875,},
			{0.705882353,	0.015686275,	0.149019608,	1,}
	};
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

	transferTex.filterMode = cudaFilterModeLinear;
	transferTex.normalized = true;    // access with normalized texture coordinates
	transferTex.addressMode[0] = cudaAddressModeBorder;//cudaAddressModeClamp;   // wrap texture coordinates

	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));

	//Creating TransferTexIso
	float4 transferFuncIso[] =
	{
			{  0.0, 1.0, 0.0, 1.0 },
			{  0.0, 1.0, 0.0, 1.0 }
	};

	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArrayIso;
	checkCudaErrors(cudaMallocArray(&d_transferFuncArrayIso, &channelDesc3, sizeof(transferFuncIso)/sizeof(float4), 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArrayIso, 0, 0, transferFuncIso, sizeof(transferFuncIso), cudaMemcpyHostToDevice));

	transferTexIso.filterMode = cudaFilterModeLinear;
	transferTexIso.normalized = true;    // access with normalized texture coordinates
	transferTexIso.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates

	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(transferTexIso, d_transferFuncArrayIso, channelDesc3));





}

void freeCudaBuffers()
{
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}
//void initCudaCubicSurface(const ushort* voxels, uint3 volumeSize)
void initCudaCubicSurface(const uchar* voxels, uint3 volumeSize)
{

	// calculate the b-spline coefficients
	cudaPitchedPtr bsplineCoeffs = CastVolumeHostToDevice(voxels, volumeSize.x, volumeSize.y, volumeSize.z);
	CubicBSplinePrefilter3DTimer((float*)bsplineCoeffs.ptr, (uint)bsplineCoeffs.pitch, volumeSize.x, volumeSize.y, volumeSize.z);

	// create the b-spline coefficients texture
	cudaArray *coeffArray = 0;
	cudaExtent volumeExtent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
	CreateTextureFromVolume(&coeffs, &coeffArray, bsplineCoeffs, volumeExtent, true);
	//    CUDA_SAFE_CALL(cudaFree(bsplineCoeffs.ptr));  //they are now in the coeffs texture, we do not need this anymore
	cudaFree(bsplineCoeffs.ptr);
	// Now create a texture with the original sample values for nearest neighbor and linear interpolation
	// Note that if you are going to do cubic interpolation only, you can remove the following code

	CreateTextureFromVolume(&tex_cubic, &volumeArray, voxels, volumeExtent, false);
	tex_cubic.addressMode[0] = cudaAddressModeBorder;
	tex_cubic.addressMode[1] = cudaAddressModeBorder;
	tex_cubic.addressMode[2] = cudaAddressModeBorder;




}

__device__ float max( float value )
{
	if( value < 0.0 )
		return 0.0;
	else
		return value;
}

__device__ int randomNumber()
{
	//	return rand()%7;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ float4 bisection(float3 start, float3 next,float3 direction, float stepSize, float isoValue)
{
	float tstep = stepSize/2;
	float3 a= start;
	float3 b = start+direction*tstep;
	float3 c = next;
	float3 point;
	float val = 0.0f;
	float temp_a = tex3D(tex, a.x , a.y , a.z ) - isoValue;
	float temp_b = tex3D(tex, b.x , b.y , b.z ) - isoValue;
	float temp_c = tex3D(tex, c.x , c.y , c.z ) - isoValue;
	int count = 0;
	float4 sample = make_float4(0.0f);

	while(count<25)
	{

		if(fabs(temp_b) <= (1e-6))
		{
			break;
		}

		if(temp_a*temp_b < 0)
		{
			tstep = tstep/2;
			c = b;
			b = a + direction * tstep;
		}
		else if(temp_b * temp_c < 0)
		{
			a = b;
			tstep = (3/4)*stepSize;
			b = a + direction*tstep;
		}
		val = tex3D(tex, b.x , b.y , b.z );
		point = b;
		if(fabs(val - isoValue)<= (1e-6))
		{
			break;
		}
		count++;
	}

	/*
         while(count<25)
    {
        if(fabs(temp_b) <= (1e-6))
        {
            break;
        }
        if(temp_a*temp_b < 0)
        {
            tstep = tstep/2;
            c = b;
            b = a + direction * tstep;
        }
        else if(temp_b * temp_c < 0)
        {
            a = b;
            tstep = (3/4)*stepSize;
            b = a + direction*tstep;
        }
        val = tex3D(tex, b.x , b.y , b.z );
        point = b;
        if(fabs(val - isoValue)<= (1e-6))
        {
            break;
        }
        count++;
    }
	 */
	sample.w = val;
	sample.x = b.x;
	sample.y = b.y;
	sample.z = b.z;

	return sample;
}

__global__ void generateAddress_One(int *variance, int *d_X, int *d_Y, int *d_linPattern)
{

	__shared__ int pattern[TILE_H*TILE_W];
	int GW = gridDim.x * blockDim.x + (gridDim.x + 1) * PAD;
	int GH = gridDim.y * blockDim.y + (gridDim.y + 1) * PAD;
	int STRIPSIZE = GW * (blockDim.y + PAD);

	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = x + y * GW;
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;

	d_linPattern[haloIndex] = TenPercent[localIndex];
	//	d_X[haloIndex] = haloIndex%GW;
	//	d_Y[haloIndex] = haloIndex/GW;

	/*
	if(variance[bid] == 1)
	{
		linear[haloIndex] = TenPercent[localIndex];
//		linear[index] = index;
		return;
	}

	else if(variance[bid] == 2)
	{
		linear[haloIndex] =TwentyPercent[localIndex] || TenPercent[localIndex];
//		linear[index] = index;
		return;
	}
	else if(variance[bid] == 3)
	{
		linear[haloIndex] = ThirtyPercent[localIndex] || TenPercent[localIndex];
//		linear[index] = index;
		return;
	}
	 */

}

__global__ void generateAddress_Two(int *variance, int *X, int *Y, int *linear)
{

	__shared__ int pattern[TILE_H*TILE_W];
	int GW = gridDim.x * blockDim.x + (gridDim.x + 1) * PAD;
	int GH = gridDim.y * blockDim.y + (gridDim.y + 1) * PAD;
	int STRIPSIZE = GW * (blockDim.y + PAD);

	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = x + y * GW;
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;

	if(variance[bid] == 4)
	{
		linear[haloIndex] = FourtyPercent[localIndex]|| TenPercent[localIndex];
		return;
	}

	else if(variance[bid] == 5)
	{
		linear[haloIndex] =FiftyPercent[localIndex]|| TenPercent[localIndex];
		return;
	}
	else if(variance[bid] == 6)
	{
		linear[haloIndex] = SixtyPercent[localIndex]|| TenPercent[localIndex];
		return;
	}

}

__global__ void generateAddress_Three(int *variance, int *X, int *Y, int *linear)
{

	__shared__ int pattern[TILE_H*TILE_W];
	int GW = gridDim.x * blockDim.x + (gridDim.x + 1) * PAD;
	int GH = gridDim.y * blockDim.y + (gridDim.y + 1) * PAD;
	int STRIPSIZE = GW * (blockDim.y + PAD);

	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = x + y * GW;
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;

	if(variance[bid] == 7)
	{
		linear[haloIndex] = SeventyPercent[localIndex]|| TenPercent[localIndex];
		return;
	}

	else if(variance[bid] == 8)
	{
		linear[haloIndex] =EightyPercent[localIndex]|| TenPercent[localIndex];
		return;
	}


}


void generateAddress(dim3 grid, dim3 block, int *variance, int *d_X, int *d_Y, int *d_linPattern)
{
	cudaStreamCreate(&grOne);
	cudaStreamCreate(&grTwo);
	cudaStreamCreate(&grThree);
	generateAddress_One<<<grid,block, 0, grOne>>>(variance, d_X, d_Y, d_linPattern);
	//	generateAddress_Two<<<grid,block, 0, grTwo>>>(variance, X, Y, linear);
	//	generateAddress_Three<<<grid,block, 0, grThree>>>(variance, X, Y, linear);
	cudaStreamDestroy(grOne);
	cudaStreamDestroy(grTwo);
	cudaStreamDestroy(grThree);
	cudaDeviceSynchronize();
}


__device__ void addByReduction(volatile float *cache, float *temp)
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


__device__ void volumeRender(int tempX, int tempY, int tempLin, float *d_vol, float *d_red, float *d_green, float *d_blue, float *d_gray, float *res_red,
		float *res_green, float *res_blue, int imageW, int imageH, float density, float brightness,float transferOffset, float transferScale, bool isoSurface,
		float isoValue, bool lightingCondition, float tstep,bool cubic, bool cubicLight, int filterMethod)
{
	int maxSteps =1000;
	const float opacityThreshold = 1.00f;
	float4 backGround = make_float4(0.5f);
	float4 sum, col;
	float I = 5.5f;
	float ka = 0.25f; //0.0025f;
	float I_amb = 0.2;
	float kd = 0.7;
	float I_dif;
	float ks = 0.5;
	float I_spec;
	float phong = 0.0f;
	float tstepGrad = 0.001f;
	float4 value;
	float sample;

	float x_space, y_space, z_space, x_dim, y_dim, z_dim, xAspect, yAspect, zAspect;
	x_dim = d_vol[0];
	y_dim = d_vol[1];
	z_dim = d_vol[2];

	x_space = d_vol[3];
	y_space = d_vol[4];
	z_space = d_vol[5];

	int pixel = (int)d_vol[6];

	float3 minB = (make_float3(-x_space, -y_space, -z_space));
	float3 maxB = (make_float3(x_space, y_space, z_space));
	const float3 boxMin = minB;//make_float3(-0.9316f, -0.9316f, -0.5f);
	const float3 boxMax = maxB;//make_float3( 0.9316f, 0.9316f, 0.5f);

	float u = (tempX/(float)imageW)*2.0f - 1.0f;
	float v = (tempY/(float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -1.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit)
	{
		d_gray[tempLin] = (backGround.x + backGround.y + backGround.z)/3.0;
		d_red[tempLin] = backGround.x;
		res_red[tempLin] = backGround.x;
		d_green[tempLin] = backGround.y;
		res_green[tempLin] = backGround.y;
		d_blue[tempLin] = backGround.z;
		res_blue[tempLin] = backGround.z;
		return;
	}
	else
	{
		float grad_x, grad_y, grad_z;
		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane
		sum = make_float4(0.0f);
		// march along ray from front to back, accumulating color
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d*tnear;
		float3 step = eyeRay.d*tstep;
		col = make_float4(0.0f);
		sample = 0.0f;
		float3 next;
		float3 start, mid, end, gradPos;
		float preValue, postValue;


		bool flag = false;

		pos.x = ((pos.x/x_space) *0.5f + 0.5f);//*(x_dim/x_dim)*(x_space/x_space); //pos.x = (pos.x *0.5f + 0.5f)/x_aspect;
		pos.y = ((pos.y/y_space) *0.5f + 0.5f);//(x_dim/y_dim)*(x_space/x_space);
		pos.z = ((pos.z/z_space) * 0.5f + 0.5f);//(x_dim/z_dim)*(x_space/z_space);

		for (int i=0; i<maxSteps; i++)
		{
			if(lightingCondition)
			{
				isoSurface = false;
				cubic = false;
				sample = tex3D(tex, pos.x, pos.y, pos.z);
				//				sample *= 8.0f;
				col = tex1D(transferTex, (sample-transferOffset)*transferScale);
				gradPos.x = pos.x;
				gradPos.y = pos.y;
				gradPos.z = pos.z;

				preValue = tex3D(tex, (gradPos.x-tstepGrad), gradPos.y, gradPos.z);
				postValue = tex3D(tex, (gradPos.x+tstepGrad), gradPos.y, gradPos.z);
				grad_x = (postValue-preValue)/(2.0f*tstepGrad);

				preValue = tex3D(tex, gradPos.x, (gradPos.y-tstepGrad), gradPos.z);
				postValue = tex3D(tex, gradPos.x, (gradPos.y+tstepGrad), gradPos.z);
				grad_y = (postValue-preValue)/(2.0f*tstepGrad);

				preValue = tex3D(tex, gradPos.x, gradPos.y, (gradPos.z-tstepGrad));
				postValue = tex3D(tex, gradPos.x, gradPos.y, (gradPos.z+tstepGrad));
				grad_z = (postValue-preValue)/(2.0f*tstepGrad);

				float3 dir = normalize(eyeRay.d);
				float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
				I_dif = max(dot(norm, dir))*kd;
				float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
				float I_spec = pow(max(dot(dir, R)), 128.0f);
				phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);
				col.w *= density;
				col.x = I_amb* col.w  + clamp(col.w*col.x*(phong), 0.0, 1.0);
				col.y = I_amb* col.w  + clamp(col.w*col.y*(phong), 0.0, 1.0);
				col.z = I_amb* col.w  + clamp(col.w*col.z*(phong), 0.0, 1.0);

				sum = sum + col*pow((1.0f - sum.w),(0.004f/tstep));

			}
			else if(isoSurface)
			{
				lightingCondition = false;
				cubic = false;
				start = pos;
				next = pos + eyeRay.d*tstep;
				float3 coord;
				coord.x = start.x*x_dim;
				coord.y = start.y*y_dim;
				coord.z = start.z*z_dim;
				float temp1 = cubicTex3D(tex_cubic, coord);
				coord.x = next.x*x_dim;
				coord.y = next.y*y_dim;
				coord.z = next.z*z_dim;
				float temp2 = cubicTex3D(tex_cubic, coord);

				float val1 = temp1 - isoValue;
				float val2 = temp2 - isoValue;
				if(val1*val2<0)
				{
					value = bisection(start,next,eyeRay.d,tstep,isoValue);
					sample = value.w;
					gradPos.x = value.x;
					gradPos.y = value.y;
					gradPos.z = value.z;

					flag = true;
				}
				else if(val1 == isoValue)
				{
					sample = temp1;
					gradPos.x = start.x;
					gradPos.y = start.y;
					gradPos.z = start.z;
					flag = true;
				}
				else if(val2 == isoValue)
				{
					sample = temp2;
					gradPos.x = next.x;
					gradPos.y = next.y;
					gradPos.z = next.z;
					flag = true;
				}
				if(flag)
				{
					sum = tex1D(transferTexIso, (sample-transferOffset)*transferScale);
					preValue = tex3D(tex, (gradPos.x-tstepGrad) , gradPos.y , gradPos.z );
					postValue = tex3D(tex, (gradPos.x+tstepGrad) , gradPos.y , gradPos.z );
					grad_x = (postValue-preValue)/(2*tstepGrad);

					preValue = tex3D(tex, gradPos.x , (gradPos.y-tstepGrad) , gradPos.z );
					postValue = tex3D(tex, gradPos.x , (gradPos.y+tstepGrad) , gradPos.z );
					grad_y = (postValue-preValue)/(2*tstepGrad);

					preValue = tex3D(tex, gradPos.x , gradPos.y , (gradPos.z-tstepGrad) );
					postValue = tex3D(tex, gradPos.x , gradPos.y , (gradPos.z+tstepGrad) );
					grad_z = (postValue-preValue)/(2*tstepGrad);

					float3 dir = normalize(eyeRay.d);
					float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
					I_dif = max(dot(norm, dir))*kd;
					float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
					float I_spec = pow(max(dot(dir, R)), 128.0f);
					phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);
					sum.x = 1.0*phong;
					sum.y = 1.0*phong;
					sum.z = 1.0*phong;
					sum.w = 1;
					break;
				}

			}
			else if(cubic)
			{
				isoSurface = false;
				lightingCondition = false;


				float3 coord;
				coord.x = pos.x*x_dim;
				coord.y = pos.y*y_dim;
				coord.z = pos.z*z_dim;
				if(filterMethod == 1){
					sample = linearTex3D(tex_cubic, coord);
				}
				else if(filterMethod == 2){
					sample = cubicTex3D(tex_cubic, coord);
					//					sample *= 8.0f;
				}
				else
				{
					sample = cubicTex3D(tex_cubic, coord);
					//					sample *= 8.0f;
				}
				col = tex1D(transferTex, (sample - transferOffset)*transferScale);

				if(cubicLight)
				{
					gradPos.x = pos.x;
					gradPos.y = pos.y;
					gradPos.z = pos.z;


					preValue = cubicTex3D(tex_cubic, ((gradPos.x-tstepGrad))*x_dim, (gradPos.y)*y_dim, (gradPos.z)*z_dim);
					postValue = cubicTex3D(tex_cubic, ((gradPos.x+tstepGrad))*x_dim, (gradPos.y)*y_dim, (gradPos.z)*z_dim);
					grad_x = (postValue-preValue)/(2.0f*tstepGrad*x_dim);

					preValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, ((gradPos.y-tstepGrad))*y_dim, (gradPos.z)*z_dim);
					postValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, ((gradPos.y+tstepGrad))*y_dim, (gradPos.z)*z_dim);
					grad_y = (postValue-preValue)/(2.0f*tstepGrad*y_dim);

					preValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, (gradPos.y)*y_dim, ((gradPos.z-tstepGrad))*z_dim);
					postValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, (gradPos.y)*y_dim, ((gradPos.z+tstepGrad))*z_dim);
					grad_z = (postValue-preValue)/(2.0f*tstepGrad*z_dim);

					float3 dir = normalize(eyeRay.d);
					float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
					I_dif = max(dot(norm, dir))*kd;
					float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
					float I_spec = pow(max(dot(dir, R)), 128.0f);
					phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);
					col.w *= density;
					col.x = I_amb* col.w  + clamp(col.w*col.x*(phong), 0.0, 1.0);
					col.y = I_amb* col.w  + clamp(col.w*col.y*(phong), 0.0, 1.0);
					col.z = I_amb* col.w  + clamp(col.w*col.z*(phong), 0.0, 1.0);
				}
				else
				{
					col.w *= density;
					col.x *= col.w;
					col.y *= col.w;
					col.z *= col.w;
				}
				sum = sum + col*pow((1.0f - sum.w), (0.004f/tstep));
			}
			else
			{
				sample = tex3D(tex, pos.x, pos.y, pos.z);
				//				sample *= 8.0f;
				col = tex1D(transferTex, (sample-transferOffset)*transferScale);
				col.w *= density;
				col.x *= col.w;
				col.y *= col.w;
				col.z *= col.w;
				sum = sum + col*pow((1.0f - sum.w),(0.004f/tstep));

			}


			// exit early if opaque
			if (sum.w > opacityThreshold)
			{
				break;
			}

			t += tstep;

			if (t > tfar) break;

			pos += step;
		}

		sum = sum + backGround * (1.0f - sum.w);

		sum *= brightness;

		d_gray[tempLin] = (sum.x + sum.y + sum.z)/3.0;
		d_red[tempLin] = sum.x;
		res_red[tempLin] = sum.x;
		d_green[tempLin] = sum.y;
		res_green[tempLin] = sum.y;
		d_blue[tempLin] = sum.z;
		res_blue[tempLin] = sum.z;



	}

}


__device__ void loadPattern(int *pattern, int temp)
{/*
	switch (temp)
	{
	case 1:
		for(int i = 0; i<256; i++)
		{
			pattern[i] = TenPercent[i];
		}
		break;
	case 2:
		for(int i = 0; i<256; i++)
		{
			pattern[i] = TwentyPercent[i];
		}
		break;
	case 3:
			for(int i = 0; i<256; i++)
			{
				pattern[i] = ThirtyPercent[i];
			}
			break;
 */
	/*
	case 4:
			for(int i = 0; i<256; i++)
			{
				pattern[i] = FourtyPercent[i];
			}
			break;
	case 5:
			for(int i = 0; i<256; i++)
			{
				pattern[i] = FiftyPercent[i];
			}
			break;
	case 6:
			for(int i = 0; i<256; i++)
			{
				pattern[i] = SixtyPercent[i];
			}
			break;
	 */
	/*
	default:
		for(int i = 0; i<256; i++)
		{
			pattern[i] = NinetyPercent[i];
		}
	}
	 */

	//	return pattern;

}

/*
__global__ void d_render(int *d_varPriority, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue, float *res_red,
		float *res_green, float *res_blue, int imageW, int imageH, float density, float brightness,float transferOffset, float transferScale, bool isoSurface,
		float isoValue, bool lightingCondition, float tstep,bool cubic, bool cubicLight, int filterMethod, int *deviceLinear, int *d_X, int *d_Y, int onPixel)
{


	int x = blockIdx.x*blockDim.x + threadIdx.x;

		int id = x;// + y * imageW;
		int tempLin = deviceLinear[id];
		if(id>=onPixel)
			return;

		int tempX = d_X[id];
		int tempY = d_Y[id];

		volumeRender(tempX, tempY, tempLin, d_vol, d_red, d_green, d_blue, d_gray, res_red, res_green, res_blue, imageW, imageH, density, brightness, transferOffset,
				transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod);
}
 */


__global__ void d_renderFirst(float *d_var, int *d_varPriority, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue, float *res_red,
		float *res_green, float *res_blue, int imageW, int imageH, float density, float brightness,float transferOffset, float transferScale, bool isoSurface,
		float isoValue, bool lightingCondition, float tstep,bool cubic, bool cubicLight, int filterMethod,int *d_linPattern, int onPixel)
{


	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = x + y * imageW;
	int localIndex = threadIdx.x + threadIdx.y * blockDim.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int STRIPSIZE = imageW * (blockDim.y + PAD);
	int local_x, local_y;
	int tempX, tempY, tempLin;
//	int row = d_varPriority[bid];
	local_x = d_temp[0][localIndex].x;
	local_y = d_temp[0][localIndex].y;
	__shared__ float deviceData[TILE_W*TILE_H]; //TILE_W*TILE_H
	__shared__ float TempDeviceData[TILE_W*TILE_H];
	__shared__ float var[TILE_W*TILE_H];
	float variance;
	float sum;
	__shared__ float mean;
	//int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;
	//	if(x>=onPixel)
	//		return;
	//	else
	//	{
	deviceData[localIndex] = 0.0f;
	TempDeviceData[localIndex] = 0.0f;
	var[localIndex] = 0.0f;

	__syncthreads();
	if((local_x == 999 && local_y ==999) || localIndex>31)
	{

		return;
	}
	else
	{
		tempLin = (blockIdx.y * STRIPSIZE) + (PAD * imageW) + (local_y * imageW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + local_x;
		tempY = tempLin/imageW;
		tempX = tempLin%imageW;
		volumeRender(tempX, tempY, tempLin, d_vol, d_red, d_green, d_blue, d_gray, res_red, res_green, res_blue, imageW, imageH,
				density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod);

		deviceData[localIndex] = d_gray[tempLin];
		TempDeviceData[localIndex] = d_gray[tempLin];
		__syncthreads();
		addByReduction(TempDeviceData, &sum);
		__syncthreads();
		mean = sum/32.0;
		__syncthreads();
		var[localIndex] = (deviceData[localIndex] - mean)*(deviceData[localIndex] - mean);
		__syncthreads();

		addByReduction(var, &variance);
		__syncthreads();
		if(localIndex == 0)
		{
			d_var[bid] = variance;
			int v = (int(variance/0.20)*10)%7;
//			printf("[%d] V: %f -> %d\n", bid, variance, v);
/*			if(v>6)
			{
				v = v%7;
				printf("Bid: %d\t %d\n", bid,v);
			}*/
//			d_varPriority[bid] = 3;
//			__syncthreads();


//			printf("bidx: %d\t bidy: %d\tBID: %d var: %f\n", blockIdx.x, blockIdx.y, bid, d_varPriority[bid]);
		}
		d_linPattern[tempLin] = 1;
		d_var[tempLin] = 0.0f;

	}

}

__global__ void d_renderSecond(float *d_var, int *d_varPriority, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue, float *res_red,
		float *res_green, float *res_blue, int imageW, int imageH, float density, float brightness,float transferOffset, float transferScale, bool isoSurface,
		float isoValue, bool lightingCondition, float tstep,bool cubic, bool cubicLight, int filterMethod, int *d_linPattern, int *d_X, int *d_Y, int onPixel)
{


	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = x + y * imageW;
	int STRIPSIZE = imageW * (blockDim.y + PAD);
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * imageW) + (threadIdx.y * imageW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;
	int row = d_varPriority[bid]; //int(d_var[bid]);//
//	printf("second: %d\n", row);
//	__syncthreads();
	int local_x = d_temp[row][localIndex].x;
	int local_y = d_temp[row][localIndex].y;
	int tempX, tempY, tempLin;


	if((local_x == 999) && (local_y ==999))
	{
//		d_red[tempLin] = 0.5f;
//		d_green[tempLin] = 0.5f;
//		d_blue[tempLin] = 0.5f;
//		d_varPriority[bid] = 0;
		return;
	}
	else
	{
		tempLin = (blockIdx.y * STRIPSIZE) + (PAD * imageW) + (local_y * imageW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + local_x;
			tempY = tempLin/imageW;
			tempX = tempLin%imageW;

		volumeRender(tempX, tempY, tempLin, d_vol, d_red, d_green, d_blue, d_gray, res_red, res_green, res_blue, imageW, imageH,
				density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod);
		d_linPattern[tempLin] = 1;
//		d_varPriority[bid] = 0;
	}



}

__global__ void writePriority(int *d_varPriority)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int index = x + y * blockDim.x;
	d_varPriority[index] = index%7;

	__syncthreads();

}

__global__ void d_render_stripe(int *d_pattern, int *d_linear, int *d_xPattern, int *d_yPattern, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue, float *res_red, float *res_green, float *res_blue, int imageW, int imageH,
		float density, float brightness,float transferOffset, float transferScale, bool isoSurface, float isoValue, bool lightingCondition, float tstep,bool cubic, bool cubicLight, int filterMethod, int stripePixels)
{
	/*
	int maxSteps =1000;
	const float opacityThreshold = 1.00f;

	float4 backGround = make_float4(0.5f);
	float4 sum, col;
	float I = 5.5f;
	float ka = 0.25f; //0.0025f;
	float I_amb = 0.2;
	float kd = 0.7;
	float I_dif;
	float ks = 0.5;
	float I_spec;
	float phong = 0.0f;
	float tstepGrad = 0.001f;
	float4 value;
	float sample;

	float x_space, y_space, z_space, x_dim, y_dim, z_dim, xAspect, yAspect, zAspect;
	x_dim = d_vol[0];
	y_dim = d_vol[1];
	z_dim = d_vol[2];

	x_space = d_vol[3];
	y_space = d_vol[4];
	z_space = d_vol[5];

	int pixel = (int)d_vol[6];

	xAspect = (((x_dim - 1) * x_space)/((x_dim - 1) * x_space));
	xAspect = (((y_dim - 1) * y_space)/((x_dim - 1) * x_space));
	xAspect = (((z_dim - 1) * z_space)/((x_dim - 1) * x_space));

	//	float3 minB = (make_float3(-x_dim/x_dim, -y_dim/x_dim, -z_dim/x_dim));
	//	float3 maxB = (make_float3(x_dim/x_dim, y_dim/x_dim, z_dim/x_dim));

	float3 minB = (make_float3(-x_space, -y_space, -z_space));
	float3 maxB = (make_float3(x_space, y_space, z_space));

	const float3 boxMin = minB;//make_float3(-0.9316f, -0.9316f, -0.5f);
	const float3 boxMax = maxB;//make_float3( 0.9316f, 0.9316f, 0.5f);

	//	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	//	const float3 boxMax = make_float3( 1.0f, 1.0f, 1.0f);
	 */


	int x = blockIdx.x*blockDim.x + threadIdx.x;

	int id = x;// + y * imageW;

	if(id>=stripePixels)
		return;

	int tempLin = d_linear[id];
	int tempX = d_xPattern[id];
	int tempY = d_yPattern[id];

	volumeRender(tempX, tempY, tempLin, d_vol, d_red, d_green, d_blue, d_gray, res_red, res_green, res_blue, imageW, imageH, density, brightness, transferOffset,
			transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod);

	/*

	float u = (d_xPattern[id]/(float)imageW)*2.0f - 1.0f;
	float v = (d_yPattern[id]/(float)imageH)*2.0f - 1.0f;

//	float tempX = TenPercent[threadIdx.x + threadIdx.y * TILE_W] / TILE_W;
//	float tempY = TenPercent[threadIdx.x + threadIdx.y * TILE_W] % TILE_W;
//	float u = (tempX/(float)imageW)*2.0f - 1.0f;
//	float v = (tempY/(float)imageW)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -1.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);


	if (!hit)
	{

		d_red[tempLin] = backGround.x;
		res_red[tempLin] = backGround.x;
		d_green[tempLin] = backGround.y;
		res_green[tempLin] = backGround.y;
		d_blue[tempLin] = backGround.z;
		res_blue[tempLin] = backGround.z;
		d_gray[tempLin] = (backGround.x + backGround.y + backGround.z)/3.0;

		return;

	}
	else
	{

		float grad_x, grad_y, grad_z;


		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane
		sum = make_float4(0.0f);
		// march along ray from front to back, accumulating color
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d*tnear;
		float3 step = eyeRay.d*tstep;
		col = make_float4(0.0f);
		sample = 0.0f;
		float3 next;
		float3 start, mid, end, gradPos;
		float preValue, postValue;


		bool flag = false;

		pos.x = ((pos.x/x_space) *0.5f + 0.5f);//*(x_dim/x_dim)*(x_space/x_space); //pos.x = (pos.x *0.5f + 0.5f)/x_aspect;
		pos.y = ((pos.y/y_space) *0.5f + 0.5f);//(x_dim/y_dim)*(x_space/x_space);
		pos.z = ((pos.z/z_space) * 0.5f + 0.5f);//(x_dim/z_dim)*(x_space/z_space);

		for (int i=0; i<maxSteps; i++)
		{
			if(lightingCondition)
			{
				isoSurface = false;
				cubic = false;
				sample = tex3D(tex, pos.x, pos.y, pos.z);
				//				sample *= 8.0f;
				col = tex1D(transferTex, (sample-transferOffset)*transferScale);
				gradPos.x = pos.x;
				gradPos.y = pos.y;
				gradPos.z = pos.z;

				preValue = tex3D(tex, (gradPos.x-tstepGrad), gradPos.y, gradPos.z);
				postValue = tex3D(tex, (gradPos.x+tstepGrad), gradPos.y, gradPos.z);
				grad_x = (postValue-preValue)/(2.0f*tstepGrad);

				preValue = tex3D(tex, gradPos.x, (gradPos.y-tstepGrad), gradPos.z);
				postValue = tex3D(tex, gradPos.x, (gradPos.y+tstepGrad), gradPos.z);
				grad_y = (postValue-preValue)/(2.0f*tstepGrad);

				preValue = tex3D(tex, gradPos.x, gradPos.y, (gradPos.z-tstepGrad));
				postValue = tex3D(tex, gradPos.x, gradPos.y, (gradPos.z+tstepGrad));
				grad_z = (postValue-preValue)/(2.0f*tstepGrad);

				float3 dir = normalize(eyeRay.d);
				float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
				I_dif = max(dot(norm, dir))*kd;
				float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
				float I_spec = pow(max(dot(dir, R)), 128.0f);
				phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);
				col.w *= density;
				col.x = I_amb* col.w  + clamp(col.w*col.x*(phong), 0.0, 1.0);
				col.y = I_amb* col.w  + clamp(col.w*col.y*(phong), 0.0, 1.0);
				col.z = I_amb* col.w  + clamp(col.w*col.z*(phong), 0.0, 1.0);

				sum = sum + col*pow((1.0f - sum.w),(0.004f/tstep));

			}
			else if(isoSurface)
			{
				lightingCondition = false;
				cubic = false;
				start = pos;
				next = pos + eyeRay.d*tstep;
				float3 coord;
				coord.x = start.x*x_dim;
				coord.y = start.y*y_dim;
				coord.z = start.z*z_dim;
				float temp1 = cubicTex3D(tex_cubic, coord);
				coord.x = next.x*x_dim;
				coord.y = next.y*y_dim;
				coord.z = next.z*z_dim;
				float temp2 = cubicTex3D(tex_cubic, coord);

				float val1 = temp1 - isoValue;
				float val2 = temp2 - isoValue;
				if(val1*val2<0)
				{
					value = bisection(start,next,eyeRay.d,tstep,isoValue);
					sample = value.w;
					gradPos.x = value.x;
					gradPos.y = value.y;
					gradPos.z = value.z;

					flag = true;
				}
				else if(val1 == isoValue)
				{
					sample = temp1;
					gradPos.x = start.x;
					gradPos.y = start.y;
					gradPos.z = start.z;
					flag = true;
				}
				else if(val2 == isoValue)
				{
					sample = temp2;
					gradPos.x = next.x;
					gradPos.y = next.y;
					gradPos.z = next.z;
					flag = true;
				}
				if(flag)
				{
					sum = tex1D(transferTexIso, (sample-transferOffset)*transferScale);
					preValue = tex3D(tex, (gradPos.x-tstepGrad) , gradPos.y , gradPos.z );
					postValue = tex3D(tex, (gradPos.x+tstepGrad) , gradPos.y , gradPos.z );
					grad_x = (postValue-preValue)/(2*tstepGrad);

					preValue = tex3D(tex, gradPos.x , (gradPos.y-tstepGrad) , gradPos.z );
					postValue = tex3D(tex, gradPos.x , (gradPos.y+tstepGrad) , gradPos.z );
					grad_y = (postValue-preValue)/(2*tstepGrad);

					preValue = tex3D(tex, gradPos.x , gradPos.y , (gradPos.z-tstepGrad) );
					postValue = tex3D(tex, gradPos.x , gradPos.y , (gradPos.z+tstepGrad) );
					grad_z = (postValue-preValue)/(2*tstepGrad);

					float3 dir = normalize(eyeRay.d);
					float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
					I_dif = max(dot(norm, dir))*kd;
					float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
					float I_spec = pow(max(dot(dir, R)), 128.0f);

					//r=d−2(d⋅n)n

					phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);


					sum.x = 1.0*phong;
					sum.y = 1.0*phong;
					sum.z = 1.0*phong;
					sum.w = 1;
					break;
				}

			}
			else if(cubic)
			{
				isoSurface = false;
				lightingCondition = false;


				float3 coord;
				coord.x = pos.x*x_dim;
				coord.y = pos.y*y_dim;
				coord.z = pos.z*z_dim;
				if(filterMethod == 1){
					sample = linearTex3D(tex_cubic, coord);
				}
				else if(filterMethod == 2){
					sample = cubicTex3D(tex_cubic, coord);
					//					sample *= 8.0f;
				}
				else
				{
					sample = cubicTex3D(tex_cubic, coord);
					//					sample *= 8.0f;
				}
				col = tex1D(transferTex, (sample - transferOffset)*transferScale);

				if(cubicLight)
				{
					gradPos.x = pos.x;
					gradPos.y = pos.y;
					gradPos.z = pos.z;


					preValue = cubicTex3D(tex_cubic, ((gradPos.x-tstepGrad))*x_dim, (gradPos.y)*y_dim, (gradPos.z)*z_dim);
					postValue = cubicTex3D(tex_cubic, ((gradPos.x+tstepGrad))*x_dim, (gradPos.y)*y_dim, (gradPos.z)*z_dim);
					grad_x = (postValue-preValue)/(2.0f*tstepGrad*x_dim);

					preValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, ((gradPos.y-tstepGrad))*y_dim, (gradPos.z)*z_dim);
					postValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, ((gradPos.y+tstepGrad))*y_dim, (gradPos.z)*z_dim);
					grad_y = (postValue-preValue)/(2.0f*tstepGrad*y_dim);

					preValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, (gradPos.y)*y_dim, ((gradPos.z-tstepGrad))*z_dim);
					postValue = cubicTex3D(tex_cubic, (gradPos.x)*x_dim, (gradPos.y)*y_dim, ((gradPos.z+tstepGrad))*z_dim);
					grad_z = (postValue-preValue)/(2.0f*tstepGrad*z_dim);

					float3 dir = normalize(eyeRay.d);
					float3 norm = normalize(make_float3(grad_x, grad_y,grad_z));
					I_dif = max(dot(norm, dir))*kd;
					float3 R = normalize(dir + (2.0 * dot(dir,norm)*norm));
					float I_spec = pow(max(dot(dir, R)), 128.0f);
					phong = clamp(I_dif + I_spec+ ka * I_amb, 0.0, 1.0);
					col.w *= density;
					col.x = I_amb* col.w  + clamp(col.w*col.x*(phong), 0.0, 1.0);
					col.y = I_amb* col.w  + clamp(col.w*col.y*(phong), 0.0, 1.0);
					col.z = I_amb* col.w  + clamp(col.w*col.z*(phong), 0.0, 1.0);



				}
				else
				{
					col.w *= density;
					col.x *= col.w;
					col.y *= col.w;
					col.z *= col.w;

				}

				sum = sum + col*pow((1.0f - sum.w), (0.004f/tstep));



			}
			else
			{


				sample = tex3D(tex, pos.x, pos.y, pos.z);
				//				sample *= 8.0f;
				col = tex1D(transferTex, (sample-transferOffset)*transferScale);
				col.w *= density;
				col.x *= col.w;
				col.y *= col.w;
				col.z *= col.w;

				sum = sum + col*pow((1.0f - sum.w),(0.004f/tstep));

			}


			// exit early if opaque
			if (sum.w > opacityThreshold)
			{
				break;
			}

			t += tstep;

			if (t > tfar) break;

			pos += step;
		}

		sum = sum + backGround * (1.0f - sum.w);

		sum *= brightness;

		d_red[tempLin] = sum.x;
		res_red[tempLin] = sum.x;
		d_green[tempLin] = sum.y;
		res_green[tempLin] = sum.y;
		d_blue[tempLin] = sum.z;
		res_blue[tempLin] = sum.z;
		d_gray[tempLin] = (sum.x + sum.y + sum.z)/3.0;
	}


	 */
}




void render_kernel(dim3 gridVol, dim3 gridVolStripe, dim3 blockSize, float *d_var, int *d_varPriority, int *d_pattern, int *d_linear, int *d_xPattern, int *d_yPattern, float *d_vol, float *d_gray, float *d_red, float *d_green, float *d_blue,
		float *res_red, float *res_green, float *res_blue, float *device_x, float *device_p, int imageW, int imageH, float density, float brightness, float transferOffset,
		float transferScale,bool isoSurface, float isoValue, bool lightingCondition, float tstep, bool cubic, bool cubicLight, int filterMethod, int *d_linPattern, int *d_X, int *d_Y, int onPixel, int stripePixels)
{
	//	cudaEventCreate(&start);
	//	cudaEventRecord(start,0);
	cudaStreamCreate(&stripe);
	cudaStreamCreate(&blocks);
	/*
	render_kernel(gridVol, gridRender, blockSize, d_varPriority, d_pattern, d_linear, d_xPattern, d_yPattern, d_vol,d_gray, d_red, d_green, d_blue, res_red, res_green, res_blue, device_x, device_p,
	       			width, height, density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod,d_temp,
	       			deviceLinear, d_X, d_Y, onPixel);
	 */

	d_render_stripe<<<gridVolStripe, 256, 0, stripe>>>(d_pattern, d_linear, d_xPattern, d_yPattern, d_vol, d_gray, d_red, d_green, d_blue,res_red, res_green, res_blue,
			imageW, imageH, density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, stripePixels);


	d_renderFirst<<<gridVol, blockSize, 0, blocks>>>(d_var, d_varPriority, d_vol, d_gray, d_red, d_green, d_blue,res_red, res_green, res_blue, imageW, imageH, density, brightness,
			transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, d_linPattern, onPixel);


	d_renderSecond<<<gridVol,blockSize, 0, blocks>>>(d_var, d_varPriority, d_vol, d_gray, d_red, d_green, d_blue, res_red, res_green, res_blue, imageW, imageH, density, brightness, transferOffset, transferScale,
			isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, d_linPattern, d_X, d_Y, onPixel);

	/*
	firstPass(gridVol, d_varPriority, d_vol, d_gray, d_red, d_green, d_blue, res_red, res_green, res_blue, imageW, imageH, density, prightness, transferOffset, transferScale,
			isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, onPixel);
	 */

	/*
	gridVol = dim3(iDivUp(onPixel,16));
	d_render<<<gridVol, 256, 0, blocks>>>(d_varPriority, d_vol, d_gray, d_red, d_green, d_blue,res_red, res_green, res_blue, imageW, imageH, density, brightness,
			transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, deviceLinear, d_X, d_Y, onPixel);
	 */
	cudaStreamDestroy(stripe);
	cudaStreamDestroy(blocks);
	cudaDeviceSynchronize();


}
//d_output, d_vol, res_red, res_green, res_blue, imageW, imageH, d_xPattern, d_yPattern, d_linear
__global__ void blend(int *d_varPriority, bool reconstruct, int *d_linPattern, uint *d_output,float *d_red, float *d_green, float *d_blue, float *res_red, float *res_green, float *res_blue, int imageW, int imageH)
{

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	if((x>=imageW)||(y>=imageH))
		return;
	int index = x + y * imageW;
	float4 temp = make_float4(0.0f);

	if(reconstruct)
	{
		temp.x = res_red[index];
		temp.y = res_green[index];
		temp.z = res_blue[index];
		d_output[index] = rgbaFloatToInt(temp);
		res_red[index] = 0.0f;
		res_green[index] = 0.0f;
		res_blue[index] = 0.0f;
		d_linPattern[index] = 0;
	}
	else{
		temp.x = d_red[index];
		temp.y = d_green[index];
		temp.z = d_blue[index];
		d_output[index] = rgbaFloatToInt(temp);
		d_red[index] = 0.0f;
		d_green[index] = 0.0f;
		d_blue[index] = 0.0f;
		res_red[index] = 0.0f;
		res_green[index] = 0.0f;
		res_blue[index] = 0.0f;
		d_linPattern[index] = 0;
	}

	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	if(localIndex == 0)
	{
		d_varPriority[bid] = 0;
	}


}
//    blendFunction(gridVol, blockSize, d_output,d_vol, res_red, res_green, res_blue, height, width, d_xPattern, d_yPattern, d_linear);
void blendFunction(dim3 grid, dim3 block, int *d_varPriority, bool reconstruct, int *d_linPattern, uint *d_output, float *d_red, float *d_green, float *d_blue, float *res_red, float *res_green, float *res_blue, int imageH, int imageW)
{
	blend<<<grid, block>>>(d_varPriority, reconstruct, d_linPattern, d_output, d_red, d_green, d_blue, res_red, res_green, res_blue, imageW, imageH);
}


void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

void copyTenPercentage(int *pixels)
{
	//	checkCudaErrors(cudaMemcpyToSymbol(TenPercent, pixels, sizeof(int)*TILE_W*TILE_H));
}

void copyAllPercentageRenderer(int *ten, int *twenty, int *thirty, int *fourty, int *fifty, int *sixty, int *seventy, int *eighty)
{

	checkCudaErrors(cudaMemcpyToSymbol(TenPercent, ten, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(TwentyPercent, twenty, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(ThirtyPercent, thirty, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(FourtyPercent, fourty, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(FiftyPercent, fifty, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(SixtyPercent, sixty, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(SeventyPercent, seventy, sizeof(int)*TILE_W*TILE_H));
	checkCudaErrors(cudaMemcpyToSymbol(EightyPercent, eighty, sizeof(int)*TILE_W*TILE_H));

}



__global__ void varianceAnalysisKernel(float *data, float *output, int dataH, int dataW)
{
	__shared__ float deviceData[TILE_H*TILE_W];
	__shared__ float var[TILE_H*TILE_W];
	__shared__ float mean;

	__shared__ float variance;
	float temp;
	int GW = gridDim.x * blockDim.x + (gridDim.x + 1) * PAD;
	int GH = gridDim.y * blockDim.y + (gridDim.y + 1) * PAD;
	int STRIPSIZE = GW * (blockDim.y + PAD);

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x + y * dataW;

	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tx = x + (blockIdx.x +1) * PAD;
	int ty = y + (blockIdx.y +1)* PAD;

	int localIndex = threadIdx.x + threadIdx.y * TILE_W;
	int haloIndex = (blockIdx.y * STRIPSIZE) + (PAD * GW) + (threadIdx.y * GW) + (blockIdx.x + 1) * PAD + blockIdx.x * blockDim.x + threadIdx.x;
	if(localIndex >= TILE_H * TILE_W)
		return;

	deviceData[localIndex] = data[haloIndex];
	__syncthreads();
	//	printf("[%d] : [%d], %f\n",localIndex, haloIndex, data[haloIndex]);
	addByReduction(deviceData,&temp);
	__syncthreads();
	mean = temp/float(32);
	__syncthreads();
	var[localIndex] = (deviceData[localIndex] - mean)*(deviceData[localIndex] - mean);
	__syncthreads();
	addByReduction(var, &variance);
	__syncthreads();
	if(localIndex == 0)
	{
		output[bid] = variance;
	}
	/*
	if(localIndex == 0)
	{
		variance  = 0.0f;

		for(int i=0; i<TILE_W*TILE_H; i++)
		{
			variance = variance + var[i];
		}

		//		printf("[%d] Varinace sum: %.4f\n", bid,variance);
		variance = variance/float(TILE_W*TILE_H);
		//		printf("[%d] Varinace: %f\n", bid,variance);
		output[bid] = variance;
	}
	 */
	__syncthreads();



}

void varianceFunction(dim3 grid, dim3 block, float *input, float *output, int dataH, int dataW)
{
	varianceAnalysisKernel<<<grid,block>>>(input, output, dataH, dataW);
}

__global__ void testingConstantMemory()
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	//	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int y = x%7;

	//	printf("[%d]: %d\n",x, d_temp[0][x]);
	//	if(d_temp[2][x].x != 0 && d_temp[2][x].y != 0)
	//	{
	//		printf("[%d]: %d, %d ",x, d_temp[2][x].x, d_temp[2][x].y);
	//	}

}


void copyConstantTest(dim3 grid, dim3 block, int temp[7][256])
{
	checkCudaErrors(cudaMemcpyToSymbol(d_temp, temp, 7*256*sizeof(int)));
	testingConstantMemory<<<grid,block>>>();
}

void copyConstantTest_1(dim3 grid, dim3 block, int_2 temp[7][256])
{
	checkCudaErrors(cudaMemcpyToSymbol(d_temp, temp, 7*256*sizeof(int_2)));
	testingConstantMemory<<<grid,256>>>();
}


