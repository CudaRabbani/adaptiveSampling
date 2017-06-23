#include "header.h"
#include "kernel.h"
#include "reconstruction.h"
#include "deviceVars.h"

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

#define CHECK(call)

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD 0.30f

float *temp_red, *temp_green, *temp_blue;
int_2 patternMatrix[7][256];

inline void __cudaCheckError( const char *file, const int line )
{
//#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        //exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
//#endif

    return;
}



void writeVarianceResult(float *data)
{

	FILE *fp = fopen("textFiles/Variance.txt","w");
	if(!fp)
	{
		printf("Error in variance writing file\n");
	}
	else
	{
		for(int i= 0; i<gridSize.x * gridSize.y; i++)
		{
			fprintf(fp,"%f\n", data[i]);
		}
	}
	fclose(fp);
	printf("Variance result writing is done\n");
}

void writeTimer()
{
	FILE *timerFile;
	char path[50] = "Results/";
	char dimX[10];
	char dimY[10];
	char name[150]="";
	char stepSizes[150]="";
	char triVSlight[150]="";
	sprintf(dimY,"%d", GH);
	sprintf(dimX,"%d", GW);
	strcat(dimY,"by");
	strcat(dimY,dimX);
	strcat(path,dimY);
	strcat(path,"/"); //path = textFiles/Pattern/516by516_50/
	strcat(name,path);
//	strcat(name,"/Result/timing/");
	if(isoSurface)
	{
		strcat(name,"isoSurface/");
	}
	else if(lightingCondition && linearFiltering)
	{
		strcat(name,"triLinear/lightingOn/");
	}
	else if(!lightingCondition && linearFiltering)
	{
		strcat(name,"triLinear/lightingOff/");
	}
	else if(cubic && !cubicLight)
	{
		strcat(name,"triCubic/lightingOff/");
	}
	else if(cubic && cubicLight)
	{
		strcat(name,"triCubic/lightingOn/");
	}

	if(gt)
	{
		strcat(name,"groundTruth/");
	}
	else
	{
		strcat(name,"reconstructed/");
	}

	strcat(name,"timer.txt");
	printf("Timing file: %s\n", name);

	timerFile = fopen(name,"w");
	if(!timerFile)
	{
		printf("No timer file found\n");
	}
	fprintf(timerFile,"%d\n%f\n%f\n%f\n%f\n%f", frameCounter,volTimer, reconTimer, blendTimer, totalTime, (float(frameCounter)/totalTime)*1000);
	printf("Timer writing done\n");
	fclose(timerFile);
}
//writeOutput(frameCounter, WLight, WCubic, WgtLight, WgtTriCubic, WisoSurface, WgtIsoSurface, h_red, h_green, h_blue);
//WLinear, WCubic, WLinearLight, WCubicLight, WisoSurface;
void writeOutput(int frameNo, bool gt, bool WLinear, bool WLinearLight, bool WCubic, bool WCubicLight, bool WisoSurface, float *h_red, float *h_green, float *h_blue)
{
	FILE *R, *G, *B;
	FILE *binaryFile;
	rgb p;
	char path[50] = "Results/";
	char frame[10]="";
	char rgbFile[20]="";
	char dimX[10];
	char dimY[10];
	char trilinearPath[100]="";
	char trilinearLightOn[150]="";
	char trilinearLightOff[150]="";
	char tricubicPath[100]="";
	char tricubicLightOn[100]="";
	char tricubicLightOff[100]="";
	char isoSurfacePath[100]="";
	char file[150]="";


	sprintf(frame, "%d", frameNo);
	sprintf(dimY,"%d", GH);
	sprintf(dimX,"%d", GW);
	strcat(dimY,"by");
	strcat(dimY,dimX);
	strcat(path,dimY);
	strcat(path,"/"); //path = Results/516by516/
//	printf("Path: %s\n", path);
	strcat(rgbFile,"rgb_");
	strcat(rgbFile,frame);
	strcat(rgbFile,".bin"); //rgbFile = rgb_0.bin

	strcat(trilinearPath, path);
	strcat(trilinearPath, "triLinear/");
	strcat(trilinearLightOn, trilinearPath);
	strcat(trilinearLightOn, "lightingOn/");
	strcat(trilinearLightOff, trilinearPath);
	strcat(trilinearLightOff, "lightingOff/");


	strcat(tricubicPath,path);	// textFiles/Pattern/516by516_50/Result/
	strcat(tricubicPath,"triCubic/");	// textFiles/Pattern/516by516_50/Result/lighting/
	strcat(tricubicLightOn, tricubicPath);
	strcat(tricubicLightOn,"lightingOn/");
	strcat(tricubicLightOff, tricubicPath);
	strcat(tricubicLightOff,"lightingOff/");

	strcat(isoSurfacePath, path);
	strcat(isoSurfacePath, "isoSurface/");

	if(gt)
	{
		strcat(trilinearLightOn, "groundTruth/");
		strcat(trilinearLightOff, "groundTruth/");
		strcat(tricubicLightOn,"groundTruth/");
		strcat(tricubicLightOff,"groundTruth/");
		strcat(isoSurfacePath, "groundTruth/");
	}
	else
	{
		strcat(trilinearLightOn, "reconstructed/");
		strcat(trilinearLightOff, "reconstructed/");
		strcat(tricubicLightOn,"reconstructed/");
		strcat(tricubicLightOff,"reconstructed/");
		strcat(isoSurfacePath, "reconstructed/");
	}


	if(WLinear)
	{
		strcat(trilinearLightOff,rgbFile);
		strcat(file, trilinearLightOff);
	}
	else if(WLinearLight)
	{
		strcat(trilinearLightOn,rgbFile);
		strcat(file, trilinearLightOn);
	}
	else if(WCubic)
	{
		strcat(tricubicLightOff, rgbFile);
		strcat(file, tricubicLightOff);
	}
	else if(WCubicLight)
	{
		strcat(tricubicLightOn,rgbFile);
		strcat(file, tricubicLightOn);
	}
	else if(WisoSurface)
	{
		strcat(isoSurfacePath,rgbFile);
		strcat(file, isoSurfacePath);
	}

	printf("[writeOutput]: %s\n", file);

	binaryFile = fopen(file,"wb");
	if(!binaryFile)
	{
		printf("Binary File Error\n");
	}
	else{
		for(int i = 0; i<GW*GH; i++)
		{
			p.red = h_red[i];
			p.green = h_green[i];
			p.blue = h_blue[i];
			fwrite(&p, sizeof(p),1,binaryFile);
		}
//		printf("\n%s\nBinary file writing done\n",rgbBinFile);
	}
	fclose(binaryFile);


}




void calcuateTiming()
{
	float total = 0.0f;
	for(int i = 0; i<1000; i++)
	{
		total+=frameTimer[i];
	}
	printf("\nTime to generate 1000 frame is %.3f ms\nAverage FPS: %f\n", total, (float)frameCounter/total);
}

void writeOutputReconstruction(float *red, float *green, float *blue)
{
	char frame[20];
	char redFile[100] = "redOutRecon_";
	char greenFile[100]= "greenOutRecon_";
	char blueFile[100]= "blueOutRecon_";
	char r[150]="", g[150]="", b[150]="";
	char path[80] = "textFiles/Reconstruction/";
	sprintf(frame, "%d", frameCounter);
	strcat(redFile,frame);
	strcat(redFile,".txt");
	strcat(r,path);
	strcat(r,redFile);

	strcat(greenFile,frame);
	strcat(greenFile,".txt");
	strcat(g,path);
	strcat(g,greenFile);

	strcat(blueFile,frame);
	strcat(blueFile,".txt");
	strcat(b,path);
	strcat(b,blueFile);




	FILE *R, *G, *B;
	R = fopen(r,"w");
	G = fopen(g,"w");
	B = fopen(b,"w");

	for(int i=0; i< width* height; i++)
	{
		fprintf(R, "%f\n", red[i]);
		fprintf(G, "%f\n", green[i]);
		fprintf(B, "%f\n", blue[i]);
	}
	fclose(R);
	fclose(G);
	fclose(B);
//	printf("Output writing done for reconstruction of: %d %d\n", width, height);


}

void writeOutputVolume(float *red, float *green, float *blue)
{
	FILE *R, *G, *B;

	R = fopen("textFiles/redOutVol.txt","w");
	G = fopen("textFiles/greenOutVol.txt","w");
	B = fopen("textFiles/blueOutVol.txt","w");
	for(int i=0; i< width* height; i++)
	{
		fprintf(R, "%f\n", red[i]);
		fprintf(G, "%f\n", green[i]);
		fprintf(B, "%f\n", blue[i]);
	}
	fclose(R);
	fclose(G);
	fclose(B);
//	printf("From writeOutput: %d %d\n", width, height);

}

void loadFiles(float *in_red, float *in_green, float *in_blue)
{

//	printf("File Loading For Reconstruction\n");
	FILE *R, *G, *B;
	R = fopen("textFiles/redOutVol.txt", "r");
	G = fopen("textFiles/greenOutVol.txt", "r");
	B = fopen("textFiles/blueOutVol.txt", "r");
	for(int i= 0; i<width*height; i++)
	{
		fscanf(R, "%f", &in_red[i]);
		fscanf(G, "%f", &in_green[i]);
		fscanf(B, "%f", &in_blue[i]);
	}
//	printf("File loading done\n");

}

void loadPattern(int *h_pattern,int *h_linear, int *xPattern, int *yPattern, int gH, int gW, int stripePixels)
{
	FILE *pattern, *X, *Y, *patternInfo, *test, *linPatternInfo, *matrix2D;
	char H[5], W[5];
	char path[50] = "textFiles/Pattern/";
	char linCoord[15] = "_ptrnIdx.txt";
	char xCoord[15]= "Xcoord.txt";
	char yCoord[15]= "Ycoord.txt";
	char xFile[60] = "";
	char yFile[60] = "";
	char linFile[60]= "";
	char dimX[10];
	char dimY[10];
	char percent[10];
	sprintf(percent, "%d", percentage);
	sprintf(dimY,"%d", gH);
	sprintf(dimX,"%d", gW);
	strcat(dimY,"by");
	strcat(dimY,dimX);
	strcat(path,dimY);
	strcat(path,"_");
	strcat(path,percent);
	strcat(path,"/"); //path = textFiles/Pattern/516by516_0/
	char patternName[50] = "";
	char name[50] = "";
	char ext[5] = ".txt";
	sprintf(H, "%d", gH);
	sprintf(W, "%d", gW);
	strcat(name,H);
	strcat(name,"by");
	strcat(name,W);
	char patternFile[70] = "";
	strcat(patternFile,path);
	strcat(patternFile,name);
	strcat(patternFile,ext);
//	printf("Input: %s\n", patternFile);
	char matrix[15] = "matrix.txt";
	char matrixFile[80]="";
	strcat(matrixFile,path);
	strcat(matrixFile,name);
	strcat(matrixFile,matrix);
	printf("Matrix file: %s\n", matrixFile);
	char lin[70]="";
	strcat(lin,path);
	strcat(lin,dimY);
	strcat(lin,linCoord);
//	strcat(lin);
//	printf("Linear Pattern File: %s\n", lin);
	strcat(xFile,path);
	strcat(xFile,name);
	strcat(xFile,xCoord);
//	printf("X-COORD: %s\t", xFile);
	strcat(yFile,path);
	strcat(yFile,name);
	strcat(yFile,yCoord);
//	printf("Y-COORD: %s\n", yFile);
	char info[70] = "";
	strcat(info,path);
	strcat(info,name);
	strcat(info,"_patternInfo.txt");
	if(percentage != 0)
	{
		pattern = fopen(patternFile, "r");
		if(!pattern)
		{
			fprintf(stderr, "Error in opening pattern file\n");
		}
		else{
			printf("Loading Pattern\n");
			for(int i=0;i <gH*gW; i++)
			{
				fscanf(pattern, "%d", &h_pattern[i]);
			}
		}
		fclose(pattern);
	}

	X = fopen(xFile, "r");
	Y = fopen(yFile, "r");
	patternInfo = fopen(info, "r");
	linPatternInfo = fopen(lin,"r");
	matrix2D = fopen(matrixFile, "r");
	printf("Pattern Info: %s\n", info);
	if(!matrix2D)
	{
		printf("2D matirx pattern file error\n");
	}
	else
	{
		for(int i=0;i <gH*gW; i++)
		{
			fscanf(matrix2D, "%d", &h_pattern[i]);
		}
	}



	if(!X || !Y || !linPatternInfo)
	{
		fprintf(stderr, "Error in opening pattern file for X or Y or linear Pattern index\n");
	}
	else{
		for(int i = 0; i<stripePixels; i++)
		{
			fscanf(X, "%d", &xPattern[i]);
			fscanf(Y, "%d", &yPattern[i]);
			fscanf(linPatternInfo, "%d", &h_linear[i]);
		}
//		printf("Pattern reading for x and y coords done\n");
	}

	fclose(patternInfo);
	fclose(linPatternInfo);
	fclose(X);
	fclose(Y);

	printf("Pattern loading complete for %d by %d image\n", gH, gW);
}

void outputTest(float *test)
{
	FILE *f = fopen("textFiles/testOutput.txt","w");

	for(int i= 0; i<stripePixels; i++)
	{
		fprintf(f, "%f\n", test[i]);
	}
	fclose(f);
}

void writeLinearAddress(int *address)
{
	FILE *fp = fopen("textFiles/tmepAddress.txt","w");
//	printf("\n%d by %d\n", dataH, dataW);

	for(int i=0; i<516*516; i++)
	{
		fprintf(fp, "%d\n", address[i]);
	}
	fclose(fp);
//	printf("Address Writing done\n");
}

/*
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
*/
void computeFPS()
{

    frameCount++;
    frameCounter++;
    fpsCount++;
    char fps[256];

    fpsTimer = glutGet(GLUT_ELAPSED_TIME);
    if(fpsTimer - timerBase > 1000)
    {
    	sprintf(fps, "Volume Render: %3.1f fps", frameCount * 1000.0/(fpsTimer - timerBase));
    	glutSetWindowTitle(fps);
    	timerBase = fpsTimer;
    	frameCount = 0;
    }


/*
    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Volume Render: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
*/
}

int * loadPatternBlock()
{
	FILE *ten = fopen("blockPattern/10.txt","r");
	int *pixels;
	int counter = 0;
	int *temp;
	temp = (int *)malloc(sizeof(int) * 256);

	for(int i = 0; i<256; i++)
	{
		fscanf(ten, "%d", &temp[i]);
		if(temp[i] == 1)
		{
			patternMatrix[0][counter].x = i%16;
			patternMatrix[0][counter].y = i/16;
			counter++;
		}
	}
	/*
	for(int i=0; i<(counter-1); i++)
	{
		printf("[%d] : %d,%d\n", i, patternMatrix[0][i].x,patternMatrix[0][i].y);
	}
	*/
//	printf("\nTotal turned on pixel: %d\n", counter);
	return temp;

}


void adaptiveSample2D(int *in)
{

	int n = 32;
	int counter = 0;
	int blockX = 16, blockY = 16;
	int G1 = 19;
	int G2 = 28;
	int inc = G2- G1;
	int x = 0, y = 0;
	int index = 0;
	int *temp;
	int_2 matrix[7][256];
	temp = (int *)malloc(sizeof(int)*256);
	for(int i=0; i<256;i++)
	{
		temp[i] = 0;
	}
	for(int i= 1; i<7; i++)
	{
		int rowElementCounter = 0;
		counter = 0;
		while(counter < ((i+1)*n))
		{
			if(x<blockX && y<blockY)
			{

				index = x + y * blockX;
				if(in[index]==0)
				{
					temp[index] = 1;
					patternMatrix[i][rowElementCounter].x = x;
					patternMatrix[i][rowElementCounter].y = y;

					counter++;
					rowElementCounter++;
				}
			}
			x = (x + inc)%G1;
			y = (y + inc)%G2;
		}
//		printf("Adaptive Sampling 2D: %d\n", rowElementCounter);

	}


}

int * adaptiveSample(int *in, int percentage)
{
	int n = percentage;
	printf("From adaptive Sample: %d\n", n);
	int counter = 0;
	int blockX = 16, blockY = 16;
	int G1 = 19;
	int G2 = 28;
	int inc = G2- G1;
	int x = 0, y = 0;
	int index = 0;
	int *temp;
	temp = (int *)malloc(sizeof(int)*256);
	for(int i=0; i<256;i++)
	{
		temp[i] = 0;
	}
	/*
	for(int i=0; i<blockY; i++)
	{
		for(int j=0; j<blockX; j++)
		{
			index = j + i * blockX;
			temp[index] = in[index];
		}
	}
	*/

	while(counter < n)
	{
		if(x<blockX && y<blockY)
		{

			index = x + y * blockX;
			if(in[index] == 0)
			{
				temp[index] = 1;
				counter++;
			}
		}
		x = (x + inc)%G1;
		y = (y + inc)%G2;
	}

	return temp;
}

void extractAddress(int *linear)
{
	int counter = 0;
	for(int i = 0; i<width*height; i++)
	{
		if(linear[i] == 1)
		{
			host_linear[counter] = i;
			h_X[counter] = i%width;
			h_Y[counter] = i/width;
			counter++;
		}
	}

	onPixel = counter-1;
//	printf("Turned on pixel: %d\n", onPixel);
//	printf("[%d by %d]\tExtract Address:[On] %d\t[Counter]: %d\n",width, height, onPixel, counter);
}

void copyAddress()
{

	hostX = (int*)malloc(sizeof(int)*onPixel);
	hostY = (int*)malloc(sizeof(int)*onPixel);
	hostLinear = (int*)malloc(sizeof(int)*onPixel);
//	cudaMalloc(&deviceLinear, sizeof(int)*onPixel);
//	cudaMalloc(&d_X, sizeof(int)*onPixel);
//	cudaMalloc(&d_Y, sizeof(int)*onPixel);
	for(int i=0; i<onPixel; i++)
	{
		hostLinear[i] = host_linear[i];
		hostX[i] = h_X[i];
		hostY[i] = h_Y[i];
	}
//	printf("%d\t%d\t%d\n", hostLinear[onPixel-1],hostX[onPixel-1],hostY[onPixel-1]);
	cudaMemcpy(deviceLinear, hostLinear, sizeof(int)*onPixel, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X, hostX, sizeof(int)*onPixel, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, hostY, sizeof(int)*onPixel, cudaMemcpyHostToDevice);

	free(hostX);
	free(hostY);
	free(hostLinear);

//	printf("copyAddress: %d\n", onPixel);
}

// render image using CUDA
void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));




    if(frameCounter<=100)
    {
    	if(writeMode)
    	{
    	    cudaMemcpy(h_red,res_red, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    	    cudaMemcpy(h_green,res_green, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    	    cudaMemcpy(h_blue,res_blue, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    		//void writeOutput(int frameNo, bool WLinear, bool WLinearLight, bool WCubic, bool WCubicLight, bool WisoSurface, float *h_red, float *h_green, float *h_blue)
    		writeOutput(frameCounter, gt, WLinear, WLinearLight, WCubic, WCubicLight, WisoSurface, h_red, h_green, h_blue);
    	}
    }
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(float)));
    cudaEventRecord(blendStart, 0);
    blendFunction(gridBlend, blockSize, d_varPriority, reconstruct, d_linPattern, d_output, d_red, d_green, d_blue, res_red, res_green, res_blue, height, width);
    cudaEventRecord(blendStop, 0);
    cudaEventSynchronize(blendStop);
    cudaEventElapsedTime(&blendTimer, blendStart, blendStop);
//    printf("Blend time: %f ms\n",blendTimer);
//    frameCounter++;
    cudaDeviceSynchronize();
    CudaCheckError();
 /*
    void reconstructionFunction(dim3 grid, dim3 block, float *red, float *green, float *blue, int *pattern, float *red_res, float *green_res, float *blue_res,
     		int dataH, int dataW, float *device_x, float *device_p);
*/
/*
    if(frameCounter<1000)
    {
        float totalTime = volTimer + reconTimer + blendTimer;
        frameTimer[frameCounter] = totalTime;
    }
*/
    totalTime += volTimer + reconTimer + blendTimer;

//    printf("Total Time: %f ms\nTotal Frame : %d\nAverage time: %f ms\n", totalTime, frameCounter, frameCounter/totalTime);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    cudaDeviceSynchronize();
//	}
//	run = false;
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    //GLfloat modelView[16];
    GLfloat modelView[16] =
    {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 1.0f
        };

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
//    gluLookAt(-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); //--------------------------------------
//    glScalef(1.0, 1.0, 2.0);
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    cudaEventRecord(volStart, 0);
    render_kernel(gridFirstPass, gridVol, gridVolStripe, blockSize, gt, d_var, d_varPriority, h_var, h_varPriority, d_pattern, d_linear, d_xPattern, d_yPattern, d_vol,d_gray, d_red, d_green, d_blue, res_red, res_green, res_blue, device_x, device_p,
       			width, height, density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod, d_linPattern, d_X, d_Y, onPixel, stripePixels);
    cudaEventRecord(volStop, 0);
    cudaEventSynchronize(volStop);
    cudaEventElapsedTime(&volTimer, volStart, volStop);
    totalVolTimer +=volTimer;


//    cudaMemcpy(d_linPattern, h_linPattern, sizeof(int)*width*height, cudaMemcpyHostToDevice);
//    cudaMemcpy(h_linPattern, d_linPattern, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
//    writeLinearAddress(h_linPattern);


    cudaEventRecord(reconStart, 0);

    if(!gt)
    {
//    	if(reconstruct)
//    	{
    		reconstructionFunction(gridSize, blockSize, d_red, d_green, d_blue, d_linPattern, d_varPriority, res_red, res_green, res_blue, height, width, device_x, device_p);

//    	}
    }
//    cudaMemcpy(d_linPattern, h_linPattern, sizeof(int)*width*height, cudaMemcpyHostToDevice);
//    cudaMemcpy(h_linPattern, d_linPattern, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
//    writeLinearAddress(h_linPattern);

   	cudaEventRecord(reconStop, 0);
   	cudaEventSynchronize(reconStop);
   	cudaEventElapsedTime(&reconTimer, reconStart, reconStop);
   	totalReconTimer += reconTimer;


   	render();

    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    std::vector<GLubyte> emptyData(width * height * 4, 0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, &emptyData[0]);

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    if(writeMode)
    {
    	glPushMatrix();
    	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
    	glRotatef(viewRotation.y, 0.0f, 1.0f, 0.0f);
    	glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    	glPopMatrix();
    }


    float ratio =  (float)width  / (float)height;

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();
    if(writeMode)
    {
    	viewRotation.y += 1.0f;
    }

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

#endif

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            #if defined (__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                printf("\nTotal number of generated frame is: %d\nTotal time is: %f ms\nFPS: %.3f\n", frameCounter, totalTime, float(frameCounter)/totalTime*1000);
                printf("Time for volume rendering: %f ms\t FPS for volumer: %f\n", totalVolTimer, float(frameCounter)/totalVolTimer*1000);
                printf("Time for reconstruction: %f ms\t FPS for reconstruction: %f\n", totalReconTimer, float(frameCounter)/totalReconTimer*1000);
                if(writeMode)
                {
                    writeTimer();
                }
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;
        case 'r':
        	reconstruct = !reconstruct;
        	break;

        case '+':
            density += 0.01f;
            printf("Density: %f\n", density);
            break;

        case '-':
            density -= 0.01f;
            printf("Density: %f\n", density);
            break;

        case ']':
            brightness += 0.1f;
            printf("Brightness: %f\n", brightness);
            break;

        case '[':
            brightness -= 0.1f;
            printf("Brightness: %f\n", brightness);
            break;

        case ';':
            transferOffset += 0.01f;
            printf("TransferOffset: %f\n", transferOffset);
            break;

        case '\'':
            transferOffset -= 0.01f;
            printf("TransferOffset: %f\n", transferOffset);
            break;

        case '.':
            transferScale += 0.01f;
            printf("TransferScale: %f\n", transferScale);
            break;

        case ',':
            transferScale -= 0.01f;
            printf("TransferScale: %f\n", transferScale);
            break;

        case 'l':
        	lightingCondition = !lightingCondition;
        	break;

        case 'i':
        	isoSurface = !isoSurface;
        	break;
        case '>':
        	isoValue += 0.005f;
        	printf("Iso-Value: %f\n", isoValue);
        	break;
        case '<':
        	isoValue -= 0.005f;
        	printf("Iso-Value: %f\n", isoValue);
        	break;
        case 'S':
        	tstep += 0.00005f;
        	printf("Step Size: %f\n", tstep);
        	break;
        case 's':
        	tstep -= 0.00005f;
        	printf("Step Size: %f\n", tstep);
        	break;
        case 'Q':
        	cubic = !cubic;
        	break;
        case 'q':
        	cubicLight = !cubicLight;
        	break;
        case '1':
        	filterMethod = 1;
        	break;
        case '2':
			filterMethod = 2;
			break;
        default:
            break;
    }

//    printf("dens = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f, isoValue: %.3f \n", density, brightness, transferOffset, transferScale, isoValue);
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
        printf("Translation: %f\n", viewTranslation.z);
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
        printf("Rotation: %f, %f\n", viewRotation.x, viewRotation.y);
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    width = w;
    height = h;

    float newWidth = (float)w;
    float newHeight = (float)h;

    float ratio =  newWidth  / newHeight;
    ratio = 1/ratio;
    initPixelBuffer();

    gridSize = dim3(blocksX, blocksY);
    gridVolStripe = dim3(iDivUp(stripePixels,256));
//    gridVol =dim3(iDivUp(onPixel,256));
    gridBlend = dim3(iDivUp(width,blockXsize), iDivUp(height,blockYsize));

    float temp = h/w;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -ratio, ratio, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity() ;

}

void cleanup()
{
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    //glClearColor(0.0,0.0,0.0,1.0);
    glClearColor(0.5,0.5,0.5,1.0);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }

}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif

    return data;
}

void loadKernel(float *kernel, float lambda, int length)
{

	if(length == 49)
	{
		printf("Loading 7 by 7 Kernel\n");
		kernel[0] = 0.0001;
		kernel[1] = 0.0099;
		kernel[2] = -0.0793;
		kernel[3] = -0.0280;
		kernel[4] = -0.0793;
		kernel[5] = 0.0099;
		kernel[6] = 0.0001;
		kernel[7] = 0.0099;
		kernel[8] = -0.1692;
		kernel[9] = 0.6540;
		kernel[10] = 1.0106;
		kernel[11] = 0.6540;
		kernel[12] = -0.1692;
		kernel[13] = 0.0099;
		kernel[14] = -0.0793;
		kernel[15] = 0.6540;
		kernel[16] = 0.1814;
		kernel[17] = -8.0122;
		kernel[18] = 0.1814;
		kernel[19] = 0.6540;
		kernel[20] = -0.0793;
		kernel[21] = -0.0280;
		kernel[22] = 1.0106;
		kernel[23] = -8.0122;
		kernel[24] = 23.3926;
		kernel[25] = -8.0122;
		kernel[26] = 1.0106;
		kernel[27] = -0.0280;
		kernel[28] = -0.0793;
		kernel[29] = 0.6540;
		kernel[30] = 0.1814;
		kernel[31] = -8.0122;
		kernel[32] = 0.1814;
		kernel[33] = 0.6540;
		kernel[34] = -0.0793;
		kernel[35] = 0.0099;
		kernel[36] = -0.1692;
		kernel[37] = 0.6540;
		kernel[38] = 1.0106;
		kernel[39] = 0.6540;
		kernel[40] = -0.1692;
		kernel[41] = 0.0099;
		kernel[42] = 0.0001;
		kernel[43] = 0.0099;
		kernel[44] = -0.0793;
		kernel[45] = -0.0280;
		kernel[46] = -0.0793;
		kernel[47] = 0.0099;
		kernel[48] = 0.0001;
	}
	else
	{
		printf("Loading 5 by 5 Kernel\n");
		kernel[0] = 0.0f;
		kernel[1] = 4.0f;
		kernel[2] = 0.0f;
		kernel[3] = 4.0f;
		kernel[4] = 0.0f;
		kernel[5] = 4.0f;
		kernel[6] = -16.0f;
		kernel[7] = -8.0f;
		kernel[8] = -16.0f;
		kernel[9] = 4.0f;
		kernel[10] = 0.0f;
		kernel[11] = -8.0f;
		kernel[12] = 64.0f;
		kernel[13] = -8.0f;
		kernel[14] = 0.0f;
		kernel[15] = 4.0f;
		kernel[16] = -16.0f;
		kernel[17] = -8.0f;
		kernel[18] = -16.0f;
		kernel[19] = 4.0f;
		kernel[20] = 0.0;
		kernel[21] = 4.0f;
		kernel[22] = 0.0f;
		kernel[23] = 4.0f;
		kernel[24] = 0.0;

	}
	/*
    kernel[0] = 0.0001;
    kernel[1] = 0.0099;
    kernel[2] = -0.0793;
    kernel[3] = -0.0280;
    kernel[4] = -0.0793;
    kernel[5] = 0.0099;
    kernel[6] = 0.0001;
    kernel[7] = 0.0099;
    kernel[8] = -0.1692;
    kernel[9] = 0.6540;
    kernel[10] = 1.0106;
    kernel[11] = 0.6540;
    kernel[12] = -0.1692;
    kernel[13] = 0.0099;
    kernel[14] = -0.0793;
    kernel[15] = 0.6540;
    kernel[16] = 0.1814;
    kernel[17] = -8.0122;
    kernel[18] = 0.1814;
    kernel[19] = 0.6540;
    kernel[20] = -0.0793;
    kernel[21] = -0.0280;
    kernel[22] = 1.0106;
    kernel[23] = -8.0122;
    kernel[24] = 23.3926;
    kernel[25] = -8.0122;
    kernel[26] = 1.0106;
    kernel[27] = -0.0280;
    kernel[28] = -0.0793;
    kernel[29] = 0.6540;
    kernel[30] = 0.1814;
    kernel[31] = -8.0122;
    kernel[32] = 0.1814;
    kernel[33] = 0.6540;
    kernel[34] = -0.0793;
    kernel[35] = 0.0099;
    kernel[36] = -0.1692;
    kernel[37] = 0.6540;
    kernel[38] = 1.0106;
    kernel[39] = 0.6540;
    kernel[40] = -0.1692;
    kernel[41] = 0.0099;
    kernel[42] = 0.0001;
    kernel[43] = 0.0099;
    kernel[44] = -0.0793;
    kernel[45] = -0.0280;
    kernel[46] = -0.0793;
    kernel[47] = 0.0099;
    kernel[48] = 0.0001;
//    kernel[49] = 0.0001;
	*/
    for(int i=0;i<length; i++)
    {
        kernel[i] = kernel[i]* lambda;
    }

    initializeConvolutionFilter(kernel, length);

}

void readAll()
{
	FILE *fp = fopen("Dimensions.txt","r");
	if(!fp)
	{
		printf("All information reading error\n");
	}
	else
	{
		fscanf(fp, "%d", &dataH);
		fscanf(fp, "%d", &dataW);
		fscanf(fp, "%d", &percentage);
		fscanf(fp, "%d", &kernelH);
		fscanf(fp, "%d", &kernelW);
	}
	printf("DataH: %d\t DataW: %d\tPercentage: %d kernel: %d by %d\n", dataH, dataW, percentage, kernelH, kernelW);
}

int main(int argc, char **argv)
{


	FILE *volumeInfo, *patternInfo;
	char volName[50];
	char patternInformation[100] = "textFiles/Pattern/";

	char H[15], W[15], P[5];

	int volXdim, volYdim, volZdim; //Volume Size in each directions
    char *ref_file = NULL;
    float x_spacing, y_spacing, z_spacing;
    float *kernel;

    float lambda = 0.01f;
    run = true;
    frameCounter = 0;


    readAll();

    int pad = kernelH/2;
    printf("\nPad: %d\n", pad);
	blockXsize = 16;
	blockYsize = 16;
//	tenP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	twentyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	thirtyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	fortyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	fiftyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	sixtyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	seventyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	eightyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
//	ninetyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);

	for(int i=0;i<7;i++)
	{
		for(int j=0; j<256; j++)
		{
			patternMatrix[i][j].x = 999;
			patternMatrix[i][j].y = 999;
		}
	}

	tenP = loadPatternBlock();
	adaptiveSample2D(tenP);
	copyConstantTest_1( 1, 256, patternMatrix);
	copyConstantTestReconstruction(1, 256, patternMatrix);


	float beforeCeilX = ((float)(dataW-pad)/(float)(blockXsize + pad));
	float beforeCeilY = ((float)(dataH-pad)/(float)(blockYsize + pad));
	float blocksXFloat = (ceil(beforeCeilX));
	float blocksYFloat = (ceil(beforeCeilY));
	blocksX = (int)blocksXFloat;
	blocksY = (int)blocksYFloat;

    blockSize = dim3(blockXsize, blockYsize); //16,16
    gridSize = dim3(blocksX, blocksY);
	printf("No of blocks for reconstruction: %d by %d\n", blocksX, blocksY);

	GW = blocksX * blockXsize + (blocksX + 1) * pad;
	GH = blocksY * blockYsize + (blocksY + 1) * pad;
	width = GW;
	height = GH;
    int lengthOfDatainFloat = GW * GH * sizeof(float);
    int lengthOfDatainInt = GH * GW * sizeof(int);
//	onPixel = blocksX * blocksY * 32;

    printf("Window Size is: %d by %d\n", GW,GH);
    sprintf(H,"%d", GH);
    sprintf(W,"%d", GW);
    sprintf(P,"%d", percentage);
    strcat(patternInformation,H);
    strcat(patternInformation,"by");
    strcat(patternInformation,W);
    strcat(patternInformation,"_");
    strcat(patternInformation,P);
    strcat(patternInformation,"/");
    strcat(patternInformation,H);
    strcat(patternInformation,"by");
    strcat(patternInformation,W);

    strcat(patternInformation,"_patternInfo.txt");
    patternInfo = fopen(patternInformation,"r");
    fscanf(patternInfo, "%d", &stripePixels); //total number of active pixels
    printf("Using pixels: %d\nPath: %s\n", stripePixels,patternInformation);

// Pattern for stripes---------------------------------------------//
    h_linear = (int *) malloc(sizeof(int)*stripePixels);
    xPattern = (int *)malloc(sizeof(int) * stripePixels);
    yPattern = (int *)malloc(sizeof(int) * stripePixels);
    h_pattern = (int*)malloc(lengthOfDatainInt);
    cudaMalloc(&d_linear, sizeof(int)*stripePixels);
    cudaMalloc(&d_xPattern, sizeof(float) * stripePixels);
    cudaMalloc(&d_yPattern, sizeof(float) * stripePixels);
    if(cudaMalloc(&d_pattern, lengthOfDatainInt) != cudaSuccess)
    {
    	printf("cudaMalloc error for d_pattern");
    }
    printf("Total Number of Pixel is : %d\n", stripePixels);

    loadPattern(h_pattern,h_linear, xPattern, yPattern, GH, GW, stripePixels);

    cudaMemcpy(d_xPattern, xPattern, sizeof(int) * stripePixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPattern, yPattern, sizeof(int) * stripePixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear, h_linear, sizeof(int) * stripePixels, cudaMemcpyHostToDevice);
    if(cudaMemcpy(d_pattern, h_pattern, sizeof(int) * GH * GW, cudaMemcpyHostToDevice) != cudaSuccess) //h_Pattern    if(cudaMemcpy(d_pattern, h_pattern, lengthOfDatainInt, cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	printf("cudaMemcpy error for h_pattern\n");
    	return -1;
    }


//    gridVol = dim3(iDivUp(GW,blockXsize), iDivUp(GH,blockYsize));

//    gridVol = dim3(iDivUp(ceil(sqrt(stripePixels)),blockXsize), iDivUp(ceil(sqrt(stripePixels)),blockYsize));
    gridVolStripe = dim3(iDivUp(stripePixels,256));
    printf("Number of thread launched for stripe Pixels: %d\n", gridVolStripe.x*256);

    gridBlend = dim3(iDivUp(width,blockXsize), iDivUp(height,blockYsize));
//    gridSize = dim3(iDivUp(stripePixels, blockXsize), iDivUp(stripePixels, blockYsize));
    printf("Reconstruction Block: %d by %d\n", gridSize.x, gridSize.y);


    //memory allocation goes here


    in_red = (float *)malloc(lengthOfDatainFloat);
    in_green = (float *)malloc(lengthOfDatainFloat);
    in_blue = (float *)malloc(lengthOfDatainFloat);
    temp = (float*)malloc(lengthOfDatainFloat); //testing
    temp_red = (float *)malloc(lengthOfDatainFloat);
    temp_green = (float *)malloc(lengthOfDatainFloat);
    temp_blue = (float *)malloc(lengthOfDatainFloat);
    h_red = (float *)malloc(lengthOfDatainFloat);
    h_green = (float *)malloc(lengthOfDatainFloat);
    h_blue = (float *)malloc(lengthOfDatainFloat);
    h_gray = (float *)malloc(lengthOfDatainFloat);

    h_vol = (float *)malloc(sizeof(float)*7); //6 for vol->height,width,depth,x,y,z space, stripePixels
    cudaMalloc(&d_vol, sizeof(float)*7);
    cudaMalloc(&d_temp, sizeof(float)*stripePixels);
    cudaMalloc(&d_red, lengthOfDatainFloat);
    cudaMalloc(&d_green, lengthOfDatainFloat);
    cudaMalloc(&d_blue, lengthOfDatainFloat);
    cudaMalloc(&d_opacity, lengthOfDatainFloat);
    cudaMalloc(&res_red, lengthOfDatainFloat);
	cudaMalloc(&res_green, lengthOfDatainFloat);
	cudaMalloc(&res_blue, lengthOfDatainFloat);
    cudaMalloc(&recon_red, lengthOfDatainFloat);
	cudaMalloc(&recon_green, lengthOfDatainFloat);
	cudaMalloc(&recon_blue, lengthOfDatainFloat);
	cudaMalloc(&res_opacity, lengthOfDatainFloat);
	cudaMalloc(&device_x,lengthOfDatainFloat);
	cudaMalloc(&device_p,lengthOfDatainFloat);
	cudaMalloc(&d_gray,lengthOfDatainFloat);

	cudaEventCreate(&volStart);
	cudaEventCreate(&volStop);
	cudaEventCreate(&reconStart);
	cudaEventCreate(&reconStop);
	cudaEventCreate(&blendStart);
	cudaEventCreate(&blendStop);



    h_temp = (float *)malloc(sizeof(float) * stripePixels );
    h_linPattern = (int *)malloc(sizeof(int) * GH * GW);
    h_X = (int *)malloc(sizeof(int) * GH * GW);
    h_Y = (int *)malloc(sizeof(int) * GH * GW);
    host_linear = (int *)malloc(sizeof(int) * GH * GW);
    cudaMalloc(&d_linPattern, sizeof(int) * GH * GW);
    for(int i=0; i<GH * GW; i++)
    {
    	h_linPattern[i] = 0;
    	temp[i] = 0.0f;
    }
    cudaMemcpy(d_linPattern, h_linPattern, sizeof(int) * GH * GW, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, temp, sizeof(float) * GH * GW, cudaMemcpyHostToDevice);
    cudaMemcpy(device_p, temp, sizeof(float) * GH * GW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_red, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gray, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_opacity, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_red, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_green, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_blue, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_opacity, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    kernel = (float *)malloc(sizeof(float) * kernelH * kernelW);
    loadKernel(kernel, lambda,kernelH*kernelW);


    // Reconstruction Testing------------------------------------------------
/*
    loadFiles(in_red, in_green, in_blue);
    cudaMemcpy(d_red, in_red, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, in_green, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, in_blue, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_red, in_red, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_green, in_green, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_blue, in_blue, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    reconstructionFunction(gridSize, blockSize, d_red, d_green, d_blue, d_pattern, res_red, res_green, res_blue, GH, GW, device_x, device_p);
    cudaMemcpy(temp_red, res_red, lengthOfDatainFloat, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_green, res_green, lengthOfDatainFloat, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_blue, res_blue, lengthOfDatainFloat, cudaMemcpyDeviceToHost);
    writeOutputReconstruction(temp_red, temp_green,temp_blue);
*/
    //Variance Testing------------------------------------------------------------
    onPixel = 32 * gridSize.x * gridSize.y;
    printf("Initial turned on pixels: %d\n", onPixel);
    gridVol = gridSize;
//    gridFirstPass = dim3(iDivUp(onPixel,blockSize.x), iDivUp(onPixel, blockSize.y));
    printf("Number of pixels turned On in first pass: %d\nGrid configuration for volume: %d by %d\n", onPixel, gridVol.x, gridVol.y);
    h_varPriority= (int *)malloc(sizeof(int) * gridVol.x * gridVol.y);
    cudaMalloc(&d_var, sizeof(float)*gridVol.x * gridVol.y);
    h_var = (float *)malloc(sizeof(float)*gridVol.x * gridVol.y);
    for(int i =0; i<gridVol.x * gridVol.y; i++)
    {
    	h_varPriority[i] = 0;
    	h_var[i] = 0.0;
    }
    if(cudaMalloc(&d_varPriority, sizeof(int)*gridVol.x * gridVol.y) != cudaSuccess)
    {
    	printf("cudaMalloc for variance priority is error\n");
    }
    if(cudaMemcpy(d_varPriority, h_varPriority, sizeof(int)*gridVol.x * gridVol.y, cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	printf("main: Variance Priority memory copy error\n");
    }
    cudaMemcpy(d_var, h_var, sizeof(int)*gridVol.x * gridVol.y, cudaMemcpyHostToDevice);


/*
    FILE *fpVar = fopen("textFiles/varianceInput.txt","r");
    for(int i=0; i<GH*GW; i++)
    {
    	fscanf(fpVar,"%f", &in_red[i]);
    }
    cudaMemcpy(res_red, in_red, sizeof(float)*GH*GW, cudaMemcpyHostToDevice);
    varianceFunction(gridSize, blockSize, res_red, d_var_r, dataH, dataW);
    if(cudaMemcpy(h_var_r,d_var_r, sizeof(float)*gridSize.x*gridSize.y, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
    	printf("%d Variance device->host copy error\n", gridSize.x*gridSize.y);
    }
    varianceAnalysis(h_var_r,h_varPriority);
    for(int i=0; i<gridSize.x*gridSize.y; i++)
    {
//    	printf("[%d]: %d\n", i, h_varPriority[i]);
    }
    writeVarianceResult(h_var_r);
*/


//WLinear, WCubic, WLinearLight, WCubicLight, WisoSurface;

    if(linearFiltering && lightingCondition)
    {
    	WLinearLight = true;
    	WLinear = false;
    	WCubic = false;
    	WCubicLight = false;
    	WisoSurface = false;

    }
    else if(linearFiltering && !lightingCondition)
    {
    	WLinear = true;
    	WLinearLight = false;
    	WCubic = false;
    	WCubicLight = false;
    	WisoSurface = false;
    }
    else if(cubic && cubicLight)
    {
    	WCubic = false;
    	WCubicLight = true;
    	WLinear = false;
    	WLinearLight = false;
    	WisoSurface = false;
    }
    else if(cubic && !cubicLight)
        {
        	WCubic = true;
        	WCubicLight = false;
        	WLinear = false;
        	WLinearLight = false;
        	WisoSurface = false;
        }
    else if(isoSurface)
    {
    	WisoSurface = true;
    	WCubic = false;
    	WCubicLight = false;
    	WLinear = false;
    	WLinearLight = false;
    }




#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);

    // parse arguments

    volumeInfo = fopen("Volume.txt","r");
    if(volumeInfo == NULL)
    {
    	printf("Error in Volume information reading\n");
    }
    else{
    	fscanf(volumeInfo, "%s", volName);
    	fscanf(volumeInfo, "%d", &volXdim);
    	fscanf(volumeInfo, "%d", &volYdim);
    	fscanf(volumeInfo, "%d", &volZdim);
    	fscanf(volumeInfo, "%f", &x_spacing);
    	fscanf(volumeInfo, "%f", &y_spacing);
    	fscanf(volumeInfo, "%f", &z_spacing);
    }

    printf("[VOL]: %s\n[X]: %d\t[Y]: %d\t[Z]: %d", volName, volXdim,volYdim,volZdim);
    printf("\tSpacing: %.3f\t %.3f\t %.3f\n", x_spacing, y_spacing, z_spacing);
    h_vol[0] = volXdim;
    h_vol[1] = volYdim;
    h_vol[2] = volZdim;
    h_vol[3] = x_spacing;
    h_vol[4] = y_spacing;
    h_vol[5] = z_spacing;
    h_vol[6] = stripePixels;

    cudaMemcpy(d_vol, h_vol, sizeof(float)*7, cudaMemcpyHostToDevice);

    char *path = volName;//sdkFindFilePath(volumeFilename, argv[0]);

    if (path == 0)
    {
        printf("Error finding file '%s'\n", volName);
        exit(EXIT_FAILURE);
    }
    cudaPitchedPtr d_volumeMem;
    volumeSize = make_cudaExtent(volXdim, volYdim, volZdim);
    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);

    initCuda(h_volume, volumeSize);
    FILE *fp = fopen(path, "rb");
    uint3 volumeSizeCubic = make_uint3(volXdim, volYdim, volZdim);
    size_t noOfVoxels = volumeSizeCubic.x * volumeSizeCubic.y * volumeSizeCubic.z;

    uchar* voxels = new uchar[noOfVoxels];
//    ushort* voxels = new ushort[noOfVoxels];
	size_t linesRead = fread(voxels, volumeSizeCubic.x, volumeSizeCubic.y * volumeSizeCubic.z, fp);
	initCudaCubicSurface(voxels, volumeSizeCubic);
    free(h_volume);

    sdkCreateTimer(&timer);

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");

    // calculate new grid size
//    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutReshapeFunc(reshape);
        glutIdleFunc(idle);

        initPixelBuffer();


#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif



        glutMainLoop();


}

