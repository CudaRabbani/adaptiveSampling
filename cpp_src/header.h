/*
 * header.h
 *
 *  Created on: Jan 30, 2017
 *      Author: reza
 */

#ifndef HEADER_H_
#define HEADER_H_


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//#include <png.h>


#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <unistd.h>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;




const int ratioH = 512;
const int ratioW = 512;

const char *sSDKsample = "Volume Rendering Project with Reconstruction";

cudaEvent_t volStart, volStop;
cudaEvent_t reconStart, reconStop;
cudaEvent_t blendStart, blendStop;

float volTimer, reconTimer, blendTimer, totalTime = 0.0f;
int frameNumber = 0;
float frameTimer[1000];


cudaExtent volumeSize;
typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;
int width, height;
int dataH, dataW;
int window_W, window_H;
int stripePixels, percentage, onPixel;
int *d_pattern, *h_pattern, *h_linear;
int *xPattern, *yPattern;
int *h_X, *h_Y, *host_linear, *hostX, *hostY, *hostLinear, *d_X, *d_Y, *deviceLinear;
int *d_xPattern, *d_yPattern, *d_linear;
float *d_red, *d_green, *d_blue, *d_opacity;
float *in_red, *in_green, *in_blue, *temp;
float *res_red, *res_green, *res_blue, *res_opacity;
float *recon_red, *recon_green, *recon_blue;
float *h_gray, *d_gray;
float *h_vol, *d_vol;
float *h_red, *h_green, *h_blue;
float *h_var_r, *h_var_g, *h_var_b;
float *d_var_r, *d_var_g, *d_var_b;
int *h_varPriority;
int *d_varPriority;

float *h_var, *d_var;
int *h_linPattern, *d_linPattern;
int *tenP, *twentyP, *thirtyP, *fortyP, *fiftyP, *sixtyP, *seventyP, *eightyP, *ninetyP;
int *d_tenP, *d_twentyP, *d_thirtyP, *d_fortyP, *d_fiftyP, *d_sixtyP, *d_seventyP, *d_eightyP, *d_ninetyP;;


float *h_temp, *d_temp;
int GW, GH;
dim3 blockSize;//(16, 16);
dim3 gridSize;
dim3 gridVolStripe, gridVol;
dim3 gridBlend;


int blocksX, blocksY, blockXsize, blockYsize;
int kernelH, kernelW;

float *device_x, *device_p;
bool run;
int frameCounter;



float3 viewRotation;// = make_float3(-100.0, -100.0,100.0f);
float3 viewTranslation = make_float3(0.0, 0.0, -3.0f); //-3.0f
float invViewMatrix[12];
int ox, oy;
int buttonState = 0;
float angle = 0.0;


float density = 1.00f;
float brightness = 1.0f;
float transferOffset = 0.00;//0.008; 0.19;//0.0f; //0.12
float transferScale = 1.0f;
bool linearFiltering = false;
float tstep = 0.005f;
float tstepGrad = 0.01f;
bool lightingCondition = false;
bool isoSurface =false;
float isoValue = 0.208 ;
bool cubic = false;
bool cubicLight = false; // for lighting inside cubic interpolation
int filterMethod = 2;
bool writeMode = false;
bool WLight, WCubic, WgtLight, WgtTriCubic, WisoSurface, WgtIsoSurface;
bool reconstruct = false;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
int frameCount = 0;
long fpsTimer = 0;
long timerBase = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

struct rgb{
	float red;
	float green;
	float blue;
};


using namespace std;



void initPixelBuffer();




#endif /* HEADER_H_ */
