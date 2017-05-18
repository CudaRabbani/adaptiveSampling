#include "header.h"
#include "kernel.h"
#include "reconstruction.h"
#include "deviceVars.h"

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

#define CHECK(call)

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD 0.30f

float *temp_red, *temp_green, *temp_blue;


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
	if(frameCounter>=1)
		return;
	FILE *fp = fopen("textFiles/redVariance.txt","w");
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
	char path[50] = "textFiles/Pattern/";
	char dimX[10];
	char dimY[10];
	char percent[10];
	char name[150]="";
	char missingPixel[150]="";
	char stepSizes[150]="";
	char triVSlight[150]="";
	sprintf(percent, "%d", percentage);
	sprintf(dimY,"%d", GH);
	sprintf(dimX,"%d", GW);
	strcat(dimY,"by");
	strcat(dimY,dimX);
	strcat(path,dimY);
	strcat(path,"_");
	strcat(path,percent);
	strcat(path,"/"); //path = textFiles/Pattern/516by516_50/
	strcat(name,path);
//	strcat(name,"/Result/timing/");
	if(isoSurface)
	{
		strcat(name,"/Result/isoSurface/timing/");
	}
	else if(lightingCondition)
	{
		strcat(name,"/Result/lighting/timing/");
	}
	else if(cubic)
	{
		strcat(name,"/Result/tricubic/timing/");
	}
/*
	strcat(missingPixel,name);
	strcat(missingPixel,"pixVsTime/timer.txt");

	strcat(stepSizes,name);
	strcat(stepSizes,"stepVsTime/timer.txt");
*/

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
void writeOutput(int frameNo, bool lightingCondition, bool triCubic, bool gtLight, bool gtTriCubic, bool WisoSurface, bool gtIsoSurface, float *h_red, float *h_green, float *h_blue)
{
	FILE *R, *G, *B;
	FILE *binaryFile;
	rgb p;
	char path[50] = "textFiles/Pattern/";
	char dimX[10];
	char dimY[10];
	char percent[10];
	char lighting[100]="";
	char groundTruth[100]="";
	char tricubic[100]="";
	char gtLighting[120]="";
	char gtCubic[120] = "";
	char red[40] = "";
	char green[40]="";
	char blue[40] = "";
	char redFile[140] = "";
	char greenFile[140]="";
	char blueFile[140] = "";
	char rgbFile[140] = "";
	char rgbLight[140] ="";
	char rgbLightGT[140] ="";
	char rgbTricubic[140] = "";
	char rgbTricubicGT[140] = "";
	char rgbIsoSurface[140] = "";
	char rgbIsoSurfaceGT[140] = "";
	char bin[50]="";
	char rgbBinFile[150]="";
	char frame[10]="";


	sprintf(frame, "%d", frameNo);
	sprintf(percent, "%d", percentage);
	sprintf(dimY,"%d", GH);
	sprintf(dimX,"%d", GW);
	strcat(dimY,"by");
	strcat(dimY,dimX);
	strcat(path,dimY);
	strcat(path,"_");
	strcat(path,percent);
	strcat(path,"/"); //path = textFiles/Pattern/516by516_50/
	strcat(red,"red_");
	strcat(red,frame);
	strcat(red,".txt");
	strcat(rgbFile, path);
	strcat(rgbFile,"Result/"); // textFiles/Pattern/516by516_50/Result/
	strcat(green,"green_");
	strcat(green,frame);
	strcat(green, ".txt");
	strcat(blue,"blue_");
	strcat(blue,frame);
	strcat(blue, ".txt");

	strcat(bin,"rgb_");
	strcat(bin,frame);
	strcat(bin,".bin");

	strcat(lighting, path);
	strcat(lighting, "Result/");
	strcat(lighting, "lighting/"); //textFiles/Pattern/516by516_50/Result/tricubic/
	strcat(gtLighting,lighting);
	strcat(gtLighting,"groundTruth/");

	strcat(rgbLight,rgbFile);	// textFiles/Pattern/516by516_50/Result/
	strcat(rgbLight,"lighting/");	// textFiles/Pattern/516by516_50/Result/lighting/
	strcat(rgbLightGT,rgbLight);
	strcat(rgbLightGT,"groundTruth/");

	strcat(tricubic, path);
	strcat(tricubic, "Result/");
	strcat(tricubic, "tricubic/");
	strcat(gtCubic, tricubic);
	strcat(gtCubic,"groundTruth/");

	strcat(rgbTricubic,rgbFile);
	strcat(rgbTricubic,"tricubic/");
	strcat(rgbTricubicGT, rgbTricubic);
	strcat(rgbTricubicGT, "groundTruth/");

	strcat(rgbIsoSurface,rgbFile);	// textFiles/Pattern/516by516_50/Result/
	strcat(rgbIsoSurface,"isoSurface/");	// textFiles/Pattern/516by516_50/Result/lighting/
	strcat(rgbIsoSurfaceGT, rgbIsoSurface);
	strcat(rgbIsoSurfaceGT, "groundTruth/");

	if(gtLight)
	{
		strcat(redFile, gtLighting);
		strcat(redFile, red);
		strcat(greenFile, gtLighting);
		strcat(greenFile, green);
		strcat(blueFile, gtLighting);
		strcat(blueFile, blue);

		strcat(rgbLightGT,bin);
		strcat(rgbBinFile,rgbLightGT);
	}
	else if(lightingCondition)
	{
		strcat(redFile, lighting);
		strcat(redFile, red);
		strcat(greenFile, lighting);
		strcat(greenFile, green);
		strcat(blueFile, lighting);
		strcat(blueFile, blue);

		strcat(rgbLight,bin);
		strcat(rgbBinFile,rgbLight);
	}
	else if(gtTriCubic)
	{
		strcat(redFile, gtCubic);
		strcat(redFile, red);
		strcat(greenFile, gtCubic);
		strcat(greenFile, green);
		strcat(blueFile, gtCubic);
		strcat(blueFile, blue);

		strcat(rgbTricubicGT,bin);
		strcat(rgbBinFile,rgbTricubicGT);
	}
	else if(triCubic)
	{
		strcat(redFile, tricubic);
		strcat(redFile, red);
		strcat(greenFile, tricubic);
		strcat(greenFile, green);
		strcat(blueFile, tricubic);
		strcat(blueFile, blue);

		strcat(rgbTricubic,bin);
		strcat(rgbBinFile,rgbTricubic);
	}
	else if(WisoSurface)
	{
		strcat(rgbIsoSurface,bin);
		strcat(rgbBinFile,rgbIsoSurface);
	}
	else if(gtIsoSurface)
	{
		strcat(rgbIsoSurfaceGT,bin);
		strcat(rgbBinFile,rgbIsoSurfaceGT);
	}
	printf("[writeOutput]: %s\n", rgbBinFile);

	binaryFile = fopen(rgbBinFile,"wb");
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
		printf("\n%s\nBinary file writing done\n",rgbBinFile);
	}
	fclose(binaryFile);

/*
	R = fopen(redFile,"w");
	G = fopen(greenFile,"w");
	B = fopen(blueFile,"w");

	if(!R || !G || !B)
	{
		printf("File writing error");
	}

	for(int i = 0; i<GW*GH; i++)
	{
		fprintf(R, "%f\n", h_red[i]);
		fprintf(G, "%f\n", h_green[i]);
		fprintf(B, "%f\n", h_blue[i]);
	}


	printf("\nWriting output done\n");
	fclose(R);
	fclose(G);
	fclose(B);
*/
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

void loadPattern(int *h_pattern,int *h_linear, int *xPattern, int *yPattern, int gH, int gW, int pixelCount)
{
	FILE *pattern, *X, *Y, *patternInfo, *test, *linPatternInfo;
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
	strcat(path,"/"); //path = textFiles/Pattern/516by516/
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
	printf("Input: %s\n", patternFile);
	char lin[70]="";
	strcat(lin,path);
	strcat(lin,dimY);
	strcat(lin,linCoord);
//	strcat(lin);
	printf("Linear Pattern File: %s\n", lin);
	strcat(xFile,path);
	strcat(xFile,name);
	strcat(xFile,xCoord);
	printf("X-COORD: %s\t", xFile);
	strcat(yFile,path);
	strcat(yFile,name);
	strcat(yFile,yCoord);
	printf("Y-COORD: %s\n", yFile);
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
	printf("Pattern Info: %s\n", info);


	if(!X || !Y || !linPatternInfo)
	{
		fprintf(stderr, "Error in opening pattern file for X or Y or linear Pattern index\n");
	}
	else{
		for(int i = 0; i<pixelCount; i++)
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

	for(int i= 0; i<pixelCount; i++)
	{
		fprintf(f, "%f\n", test[i]);
	}
	fclose(f);
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

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
			counter++;
		}
	}
	printf("\nTotal turned on pixel: %d\n", counter);
	return temp;

}

int * adaptiveSample(int *in, int percentage)
{
	int n = (percentage * blockXsize * blockYsize)/100;
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

void varianceAnalysis(float *in, int *out)
{
	float min = 99999.0f, max = -99999.0f, sum = 0.0, avg = 0.0f, division = 0.0f;
	int group = 9; //10p, 20p, 30p....90p
	int ten = 0, twenty =0, thirty = 0, fourty = 0, fifty = 0, sixty = 0, seventy = 0,
			eighty = 0, ninety = 0;
	for(int i =0; i<gridSize.x*gridSize.y; i++)
	{
		if(in[i]>=max)
		{
			max = in[i];
		}
		if(in[i]<=min)
		{
			min = in[i];
		}
	}
	/*
	for(int i =0; i<gridSize.x*gridSize.y; i++)
	{
		in[i] = in[i]/max;
		printf("norm: %f\n", in[i]);
	}
	*/
	division = (max - min)/float(group);
//		printf("division: %f\n", division);
	for(int i=0; i<gridSize.x*gridSize.y; i++)
	{
		if(in[i]>=min && in[i]<division+min)
		{
			out[i] = 1;
			ten++;
		}
		else if(in[i]>=(division+min+0.000001) && in[i]<(division*2+min))
		{
			out[i] = 2;
			twenty++;
		}
		else if(in[i]>=((division*2+min)+0.000001) && in[i]<(division*3+min))
		{
			out[i] = 3;
			thirty++;
		}
		else if(in[i]>=((division*3+min)+0.000001) && in[i]<(division*4+min))
		{
			out[i] = 4;
			fourty++;
		}
		else if(in[i]>=((division*3+min)+0.000001) && in[i]<(division*5+min))
		{
			out[i] = 5;
			fifty++;
		}
		else if(in[i]>=((division*5+min)+0.000001) && in[i]<(division*6+min))
		{
			out[i] = 6;
			sixty++;
		}
		else if(in[i]>=((division*6+min)+0.000001) && in[i]<(division*7+min))
		{
			out[i] = 7;
			seventy++;
		}
		else if(in[i]>=((division*7+min)+0.000001) && in[i]<(division*8+min))
		{
			out[i] = 8;
			eighty++;
		}
		else
		{
			out[i] = 9;
			ninety++;
		}
	}
/*
	printf("Min, Max: %f %f\n", min, max);
	printf("10: %d\t20: %d\t30: %d\n", ten, twenty, thirty);
	printf("40: %d\t50: %d\t60: %d\n", fourty, fifty, sixty);
	printf("70: %d\t80: %d\t90: %d\n", seventy, eighty, ninety);
	printf("Total block from variance analysis: %d\n", ten + twenty + thirty+
			fourty + fifty + sixty + seventy + eighty + ninety);

*/

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


    cudaMemcpy(h_red,res_red, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green,res_green, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue,res_blue, sizeof(float)*height*width, cudaMemcpyDeviceToHost);

    if(frameCounter<=100)
    {
    	if(writeMode)
    	{
    		writeOutput(frameCounter, WLight, WCubic, WgtLight, WgtTriCubic, WisoSurface, WgtIsoSurface, h_red, h_green, h_blue);
    	}
    }
    checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(float)));
    cudaEventRecord(blendStart, 0);
    blendFunction(gridBlend, blockSize, d_output,d_vol, res_red, res_green, res_blue, height, width, d_xPattern, d_yPattern, d_linear);
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
    render_kernel(gridVol, gridSize, blockSize, d_varPriority, d_pattern, d_linear, d_xPattern, d_yPattern, d_vol, d_red, d_green, d_blue, res_red, res_green, res_blue, device_x, device_p,
       			width, height, density, brightness, transferOffset, transferScale, isoSurface, isoValue, lightingCondition, tstep, cubic, cubicLight, filterMethod,d_temp);
    cudaEventRecord(volStop, 0);
    cudaEventSynchronize(volStop);
    cudaEventElapsedTime(&volTimer, volStart, volStop);

    varianceFunction(gridSize, blockSize, res_blue, d_var_r, dataH, dataW);
    if(cudaMemcpy(h_var_r, d_var_r, sizeof(float)*gridSize.x*gridSize.y, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
    	printf("Variance memory copy error\n");
    }
    varianceAnalysis(h_var_r,h_varPriority);
    if(cudaMemcpy(d_varPriority, h_varPriority, sizeof(int)*gridSize.x*gridSize.y, cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	printf("Variance Priority memory copy error\n");
    }



    cudaEventRecord(reconStart, 0);

    if(percentage != 100)
    {
    	if(reconstruct)
    	{
    		reconstructionFunction(gridSize, blockSize, d_red, d_green, d_blue, d_pattern, res_red, res_green, res_blue, height, width, device_x, device_p);
/*
    		cudaMemcpy(h_red,res_red, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_green,res_green, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_blue,res_blue, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
    		writeOutputReconstruction(h_red, h_green, h_blue);
*/
    	}
    }
   	cudaEventRecord(reconStop, 0);
   	cudaEventSynchronize(reconStop);
   	cudaEventElapsedTime(&reconTimer, reconStart, reconStop);



//   	printf("Recon time: %f ms\n", reconTimer);

   	render();
    // display results
//    glClear(GL_COLOR_BUFFER_BIT);
//    glClearColor(0.0f, 0.0f, 0.0f, 1.0);

    // draw image from PBO
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
    /*
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();
	*/

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
                printf("\nTotal number of generated frame is: %d\nAverage time is: %f ms\nFPS: %.3f\n", frameCounter, totalTime, float(frameCounter)/totalTime*1000);
//                writeTimer();
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
//        printf("Translation: %f\n", viewTranslation.z);
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
/*
    float ar_new = newWidth/newHeight;
    float ar_origin = ratioW/ratioH;
    float scale_w = (float) newWidth / (float) ratioW;
    float scale_h = (float) newHeight / (float) ratioH;
    if (ar_new > ar_origin) {
            scale_w = scale_h;
        } else {
            scale_h = scale_w;
        }
    float margin_x = (newWidth - (float)ratioW * scale_w) / 2;
	float margin_y = (newHeight - (float)ratioH * scale_h) / 2;
*/
    // calculate new grid size
//    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
    gridSize = dim3(blocksX, blocksY);
//    gridVol = dim3(iDivUp(width,blockXsize), iDivUp(height,blockYsize));
//    gridVol = dim3(iDivUp(ceil(sqrt(pixelCount)),blockXsize), iDivUp(ceil(sqrt(pixelCount)),blockYsize));
    gridVol = dim3(iDivUp(pixelCount,256));
    gridBlend = dim3(iDivUp(width,blockXsize), iDivUp(height,blockYsize));
//    gridSize = dim3(iDivUp(pixelCount, blockSize.x), iDivUp(pixelCount, blockSize.y));
//    gridVol = dim3(iDivUp(sqrt(pixelCount),blockXsize), iDivUp(sqrt(pixelCount),blockYsize));



/*
	glViewport(margin_x, margin_y, (float)ratioW*scale_w, (float)newHeight*scale_h);
//	glViewport(0, 0, (float)ratioW*scale_w, (float)newHeight*scale_h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
//	glOrtho(0, ratioW * ar_new, 0, ratioH * ar_new, 0, 1.0);
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
*/
	/*
	glViewport(0, 0, width, height);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(15.0, (float)width / height, 0.0, 1.0);
   glMatrixMode(GL_MODELVIEW);
*/

    /*
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if(ratio >=1.0)
    {
    	glOrtho(-1.0f*ratio, 1.0f*ratio,-1.0, 1.0, -1.0, 1.0);
    }
    else
    {
    	glOrtho(-1.0f, 1.0f,-1.0/ratio, 1.0/ratio, -1.0, 1.0);
    }

    //glOrtho(0.0, 1.0f,0.0, 1.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
	*/

//    glViewport(0,0,w,h);
//    printf("margin: [%.2f] [%.2f] [] []");
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
/*
    dataH = 1024;
    dataW = 1024;
    percentage = 30;
*/
    readAll();

    int pad = kernelH/2;
    printf("\nPad: %d\n", pad);
	blockXsize = 16;
	blockYsize = 16;
	tenP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	twentyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	thirtyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	fortyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	fiftyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	sixtyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	seventyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	eightyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);
	ninetyP = (int*)malloc(sizeof(int)*blockXsize*blockYsize);

	tenP = loadPatternBlock();
	twentyP = adaptiveSample(tenP, 20-10);
	thirtyP = adaptiveSample(tenP, 30-10);
	fortyP = adaptiveSample(tenP, 40-10);
	fiftyP = adaptiveSample(tenP, 50-10);
	sixtyP = adaptiveSample(tenP, 60-10);
	seventyP = adaptiveSample(tenP, 70-10);
	eightyP = adaptiveSample(tenP, 80-10);
	ninetyP = adaptiveSample(tenP, 90-10);
	copyAllPercentageRenderer(tenP, twentyP,thirtyP,fortyP,fiftyP,sixtyP,seventyP,eightyP,ninetyP);
	/*
	int t =0, tw = 0;
	for(int i=0; i<256; i++)
	{
		if(tenP[i] == 1)
		{
			t++;
		}
		if(twentyP[i] == 1)
		{
			tw++;
		}
		printf("[10p]: %d\t[20p]: %d\n", tenP[i], twentyP[i]);
	}
	printf("\n10P: %d\t20P: %d\n", t, tw);
	*/
//    kernelH = 7;
//    kernelW = 7;
	float beforeCeilX = ((float)(dataW-pad)/(float)(blockXsize + pad));
	float beforeCeilY = ((float)(dataH-pad)/(float)(blockYsize + pad));
	float blocksXFloat = (ceil(beforeCeilX));
	float blocksYFloat = (ceil(beforeCeilY));
	blocksX = (int)blocksXFloat;
	blocksY = (int)blocksYFloat;
	printf("No of blocks for reconstruction: %d by %d\n", blocksX, blocksY);

	GW = blocksX * blockXsize + (blocksX + 1) * pad;
	GH = blocksY * blockYsize + (blocksY + 1) * pad;
	width = GW;
	height = GH;

    //bool WLight, WCubic, WgtLight, WgtTriCubic;
    //WLight is for write lighting output
    //WgtLight is for ground truth of lighting output
    //WCubic is for tricubic
    //WgtTriCubic is for ground truth tricubic
    //WisoSurface, WgtIsoSurface;
    if(percentage == 100)
    {
		WCubic = false;
		WLight = false;
		WisoSurface = false;
    	if(lightingCondition)
    	{
    		WgtLight = true;
    		WgtTriCubic = false;
    		WgtIsoSurface = false;
    	}
    	else if(cubic && cubicLight)
    	{
    		WgtTriCubic = true;
			WgtLight = false;
			WgtIsoSurface = false;
    	}
    	else if(isoSurface)
    	{
    		WgtTriCubic = false;
    		WgtLight = false;
    		WgtIsoSurface = true;
    	}
    }
    else
    {
    	WgtLight = false;
    	WgtTriCubic = false;
    	WgtIsoSurface = false;
    	if(lightingCondition)
		{
			WLight = true;
			WCubic = false;
			WisoSurface = false;
		}
		else if(cubic && cubicLight)
		{
			WCubic = true;
			WLight = false;
			WisoSurface = false;
		}
		else if(isoSurface)
		{
			WCubic = false;
			WLight = false;
			WisoSurface = true;
		}
    }

    //This portion is for the reconstruction setup, Ghost height and width;

//    writeOutput(1, true, false, true);

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
    fscanf(patternInfo, "%d", &pixelCount); //total number of active pixels
    printf("Using pixels: %d\nPath: %s\n", pixelCount,patternInformation);


    h_linear = (int *) malloc(sizeof(int)*pixelCount);
    cudaMalloc(&d_linear, sizeof(int)*pixelCount);

    blockSize = dim3(blockXsize, blockYsize); //16,16
    gridSize = dim3(blocksX, blocksY);
//    gridVol = dim3(iDivUp(GW,blockXsize), iDivUp(GH,blockYsize));

//    gridVol = dim3(iDivUp(ceil(sqrt(pixelCount)),blockXsize), iDivUp(ceil(sqrt(pixelCount)),blockYsize));
    gridVol = dim3(iDivUp(pixelCount,256));

    printf("Number of thread launched: %d\n", gridVol.x*gridVol.y*blockXsize*blockYsize);

    gridBlend = dim3(iDivUp(width,blockXsize), iDivUp(height,blockYsize));
//    gridSize = dim3(iDivUp(pixelCount, blockXsize), iDivUp(pixelCount, blockYsize));
    printf("Volume Block: %d by %d\nReconstruction Block: %d by %d\n", gridVol.x , gridVol.y, gridSize.x, gridSize.y);


    //memory allocation goes here
    int lengthOfDatainFloat = GW * GH * sizeof(float);
    int lengthOfDatainInt = GH * GW * sizeof(int);

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

    h_vol = (float *)malloc(sizeof(float)*7); //6 for vol->height,width,depth,x,y,z space, pixelCount
    cudaMalloc(&d_vol, sizeof(float)*7);
    cudaMalloc(&d_temp, sizeof(float)*pixelCount);
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

	cudaEventCreate(&volStart);
	cudaEventCreate(&volStop);
	cudaEventCreate(&reconStart);
	cudaEventCreate(&reconStop);
	cudaEventCreate(&blendStart);
	cudaEventCreate(&blendStop);

    h_pattern = (int*)malloc(lengthOfDatainInt);
    if(cudaMalloc(&d_pattern, lengthOfDatainInt) != cudaSuccess)
    {
    	printf("cudaMalloc error for d_pattern");
    }

    xPattern = (int *)malloc(sizeof(int) * pixelCount);
    yPattern = (int *)malloc(sizeof(int) * pixelCount);
    h_temp = (float *)malloc(sizeof(float) * pixelCount );
    for(int i=0; i<GH*GW; i++)
    {
    	temp[i] = 0.0f;
    }
    cudaMemcpy(device_x, temp, sizeof(float) * GH * GW, cudaMemcpyHostToDevice);
    cudaMemcpy(device_p, temp, sizeof(float) * GH * GW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_red, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_opacity, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_red, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_green, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_blue, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(res_opacity, temp, lengthOfDatainFloat, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xPattern, sizeof(float) * pixelCount);
    cudaMalloc(&d_yPattern, sizeof(float) * pixelCount);

    printf("Total Number of Pixel is : %d\n", pixelCount);
    loadPattern(h_pattern,h_linear, xPattern, yPattern, GH, GW, pixelCount);

    kernel = (float *)malloc(sizeof(float) * kernelH * kernelW);
    loadKernel(kernel, lambda,kernelH*kernelW);
    cudaMemcpy(d_xPattern, xPattern, sizeof(int) * pixelCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPattern, yPattern, sizeof(int) * pixelCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_linear, h_linear, sizeof(int) * pixelCount, cudaMemcpyHostToDevice);
    if(cudaMemcpy(d_pattern, h_pattern, sizeof(int) * GH * GW, cudaMemcpyHostToDevice) != cudaSuccess) //h_Pattern    if(cudaMemcpy(d_pattern, h_pattern, lengthOfDatainInt, cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	printf("cudaMemcpy error for h_pattern\n");
    	return -1;
    }

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

    h_var_r = (float *)malloc(sizeof(float) * gridSize.x * gridSize.y);
    h_varPriority= (int *)malloc(sizeof(int) * gridSize.x * gridSize.y);
    if(cudaMalloc(&d_var_r, sizeof(float)*gridSize.x*gridSize.y) != cudaSuccess)
    {
    	printf("cudaMalloc for variance is error\n");
    }
    if(cudaMalloc(&d_varPriority, sizeof(int)*gridSize.x*gridSize.y) != cudaSuccess)
    {
    	printf("cudaMalloc for variance priority is error\n");
    }
    for(int i=0; i<gridSize.x*gridSize.y; i++)
    {
    	h_varPriority[i] = 1;
    }
    if(cudaMemcpy(d_varPriority, h_varPriority, sizeof(int)*gridSize.x*gridSize.y, cudaMemcpyHostToDevice) != cudaSuccess)
    {
    	printf("Variance Priority memory copy error\n");
    }

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
    h_vol[6] = pixelCount;

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

