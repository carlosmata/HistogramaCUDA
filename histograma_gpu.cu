#include "jpeg-compressor/jpgd.h"
#include "jpeg-compressor/stb_image.c"
#include "jpeg-compressor/timer.h"
#include <ctype.h>
#include <stdio.h>
#include <cuda_runtime.h>


__host__ 
void check_CUDA_Error(const char *mensaje)
{
	 cudaError_t error;
	 cudaDeviceSynchronize();
	 error = cudaGetLastError();
	 if(error != cudaSuccess)
	 {
	 	printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	 	printf("\npulsa INTRO para finalizar...");
	 	fflush(stdin);
		char tecla = getchar();
	 	exit(-1);
	 }
}

__device__ 
void warpReduce(volatile int* sdata, int tid) {
		//sdata[tid] = sdata[tid]; 
		sdata[tid] += sdata[tid + 16]; 
		sdata[tid] += sdata[tid +  8]; 
		sdata[tid] += sdata[tid +  4]; 
		sdata[tid] += sdata[tid +  2]; 
		sdata[tid] += sdata[tid +  1]; 
}

__global__ 
void histogram1(int *d_histo, 
				const int *d_in, 
				const int SIZE_HISTO, 
				const int no_data )
{   
    // int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int threadsPerBlock = blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ int histo[ ];

    for( int i = 0; i < SIZE_HISTO; i++)
      histo[i * threadsPerBlock + tid] = 0;
      //__syncthreads();
    
    for( int i = tid; i < no_data; i += threadsPerBlock ){
      	histo[(d_in[i] % SIZE_HISTO) * threadsPerBlock + tid]++;
    }
	
      //__syncthreads();

    for( int i = tid; i < SIZE_HISTO; i += threadsPerBlock ) {
        for( int j = 1; j < threadsPerBlock; j++ )
             histo[i * threadsPerBlock] += histo[i * threadsPerBlock+j];
        d_histo[i] = histo[i * threadsPerBlock];
        //__syncthreads();
    }
}

__global__ 
void histogram2(int *d_histo, 
				const int *d_in, 
				const int SIZE_HISTO, 
				const int no_data )
{
	int threadsPerBlock = blockDim.x;
	int tid = threadIdx.x;
	int cols = threadsPerBlock + threadsPerBlock/2;
	extern __shared__ int histo[ ];

	for( int i = 0; i < SIZE_HISTO; i++){
		histo[i * cols + tid] = 0;
	}

	if( tid < threadsPerBlock/2 ) {
		for( int i = 0; i < SIZE_HISTO; i++){
			histo[i * cols + threadsPerBlock + tid] = 0;
		}
	}
	__syncthreads();

	for( int i = tid; i < no_data; i += threadsPerBlock ){
		histo[(d_in[i]) * cols + tid]++;
	}
	__syncthreads();


	for( int i = 0; i < SIZE_HISTO; i++ ) {
		warpReduce( &histo[i * cols], tid);
		if( tid == 0 ){ 
			d_histo[i] = histo[i * cols];
		}
		__syncthreads();
	}
}

__global__ 
void histogram3(int *d_histo,       /* Histogram*/
				const int *d_in,     /* data from image, R, G or B*/
				const int SIZE_HISTO, /* 8 bytes - 256*/ 
				const int no_data)	 /* Width x Height*/
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;	// Id of thread in the total blocks
	int threadsPerBlock = blockDim.x;					// Number of threads per block
	int tid = threadIdx.x;								// Id of the Thread in the block
	int numBlocks = gridDim.x;							// Number of blocks in the grid
	int dataBlock = threadsPerBlock * numBlocks;		// Number of total threads
	extern __shared__ int histo[];	// Histogram of the block
	
	int cols = threadsPerBlock;// + threadsPerBlock/2;
	
	//Initialice the part of histagram
	for( int i = 0; i < SIZE_HISTO; i++){
		histo[i * cols + tid] = 0;
	}
	//__syncthreads();
	
	//Create the histograms
	for( int i = myId; i < no_data; i += dataBlock ){
		histo[(d_in[i]) * cols + tid]++;
	}
	//__syncthreads();
	
	//Make the reductions
	for( int i = 0; i < SIZE_HISTO; i++) {
		warpReduce( &histo[i * cols + 0], tid );
		if( tid == 0 ){
			d_histo[i + SIZE_HISTO * blockIdx.x] = histo[i * cols + 0]; //
		} 
		//__syncthreads();
	}
}

/*
__global__
void reductionHisto(const int *histoR, 	//histogram total
					const int *histoG, 	//histogram total
					const int *histoB, 	//histogram total
					int *histo_finalR,	//histogram 
					int *histo_finalG,	//histogram
					int *histo_finalB,	//histogram  
					const int numberBlocks
					const int SIZE_HISTO
					){
	__shared__ int temporalR[256];
	__shared__ int temporalG[256];
	__shared__ int temporalB[256];
	
	int myID = threadIdx.x; 
	
	for(int block = 0; i < numberOfBlocks; i++){
		temporalR[myID] += histoR[block * SIZE_HISTO + myID]; 
		temporalG[myID] += histoG[block * SIZE_HISTO + myID]; 
		temporalB[myID] += histoB[block * SIZE_HISTO + myID];
		__syncthreads(); 
	}
	
	histo_finalR[myID] = temporalR[myID];
	histo_finalG[myID] = temporalG[myID];
	histo_finalB[myID] = temporalB[myID]; 		

}*/


void histogram_secuencial(int *R, int *G, int *B, int lenght){
	int histogramR[256];
	int histogramG[256];
	int histogramB[256];
	timer tm;

	//Initialize the histrogram
	for(int i = 0; i < 256; i++){
		histogramR[i] = 0;
		histogramG[i] = 0;
		histogramB[i] = 0;
	}

	tm.start();
	//Create the histogram
	for(int i = 0; i < lenght; i++){
		histogramR[R[i]]++;
		histogramG[G[i]]++;
		histogramB[B[i]]++;
	}
	tm.stop();

	//Print the histogram
	for(int i = 0; i < 256; i++){
		printf("R[%d]G[%d]B[%d] = [%d][%d][%d]\n", i, i, i, histogramR[i], histogramG[i], histogramB[i]);
	}

	printf("Compute Secuencial time: %fms --- %fs\n", tm.get_elapsed_ms(), (tm.get_elapsed_ms() / 1000));
}

void histogram_parallel(int *R, int *G, int *B, int lenght, int numberOfBlocks){	
	int SIZE_HISTO = 256;
	int NUMBER_OF_THREADS = 32;
	int SHARED_MEMORY_SIZE = (32 + 16) * SIZE_HISTO * 4;
	int histogramR[SIZE_HISTO * numberOfBlocks];
	int histogramG[SIZE_HISTO * numberOfBlocks];
	int histogramB[SIZE_HISTO * numberOfBlocks];
	timer tm;
	
	//---------------------------------------------Verify thr CUDA device-----------------------------------------------
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
				devProps.name, (int)devProps.totalGlobalMem, 
				(int)devProps.major, (int)devProps.minor, 
				(int)devProps.clockRate);
	}
	//------------------------------------------------------------------------------------------------------------------

	//Initialize the histrogram
	for(int i = 0; i < SIZE_HISTO * numberOfBlocks; i++){
		histogramR[i] = 0;
		histogramG[i] = 0;
		histogramB[i] = 0;
	}

	// declare GPU memory pointers
	int * d_inR;
	int * d_inG;
	int * d_inB;
	int * d_histoR;
	int * d_histoG;
	int * d_histoB;

	tm.start();
	// allocate GPU memory
	cudaMalloc((void **) &d_inR, lenght * sizeof(int));
	cudaMalloc((void **) &d_histoR, SIZE_HISTO * numberOfBlocks * sizeof(int));
	cudaMalloc((void **) &d_inG, lenght * sizeof(int));
	cudaMalloc((void **) &d_histoG, SIZE_HISTO * numberOfBlocks * sizeof(int));
	cudaMalloc((void **) &d_inB, lenght * sizeof(int));
	cudaMalloc((void **) &d_histoB, SIZE_HISTO * numberOfBlocks * sizeof(int));
	
	//-------------------------------------Hist-----------------------------------------------
	// transfer the arrays to the GPU
	cudaMemcpy(d_inR, R, lenght * sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_histoR, histogramR, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inG, G, lenght * sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_histoG, histogramG, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice);			  
	cudaMemcpy(d_inB, B, lenght * sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_histoB, histogramB, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice);
	
	histogram3<<<numberOfBlocks, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoR, d_inR, SIZE_HISTO, lenght);
	check_CUDA_Error("ERROR en histograma"); 
	histogram3<<<numberOfBlocks, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoG, d_inG, SIZE_HISTO, lenght);
	check_CUDA_Error("ERROR en histograma"); 
	histogram3<<<numberOfBlocks, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoB, d_inB, SIZE_HISTO, lenght);
	check_CUDA_Error("ERROR en histograma"); 
	
	//histogram2<<<1, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoR, d_inR, SIZE_HISTO, lenght);
	//histogram2<<<1, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoG, d_inG, SIZE_HISTO, lenght);
	//histogram2<<<1, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE >>>(d_histoB, d_inB, SIZE_HISTO, lenght);
	
	//histogram1<<<1, NUMBER_OF_THREADS, (32) * SIZE_HISTO * 4 >>>(d_histoR, d_inR, SIZE_HISTO, lenght);
	//histogram1<<<1, NUMBER_OF_THREADS, (32) * SIZE_HISTO * 4 >>>(d_histoG, d_inG, SIZE_HISTO, lenght);
	//histogram1<<<1, NUMBER_OF_THREADS, (32) * SIZE_HISTO * 4>>>(d_histoB, d_inB, SIZE_HISTO, lenght);
	
	// copy back the sum from GPU
	cudaMemcpy(histogramR, d_histoR, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(histogramG, d_histoG, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(histogramB, d_histoB, SIZE_HISTO * numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	//-------------------------------------------------------------------------------------

	tm.stop();

	//Print the histogram
	for(int i = 0; i < SIZE_HISTO; i++){
		for(int j = 1; j < numberOfBlocks; j++){
			histogramR[i] += histogramR[j * SIZE_HISTO + i]; 
			histogramG[i] += histogramG[j * SIZE_HISTO + i]; 
			histogramB[i] += histogramB[j * SIZE_HISTO + i];
		}
		printf("R[%d]G[%d]B[%d] = [%d][%d][%d]\n", i, i, i, histogramR[i], histogramG[i], histogramB[i]);
	}
	
	printf("Compute Parallel time: %fms --- %fs\n", tm.get_elapsed_ms(), (tm.get_elapsed_ms() / 1000));
}


// Test JPEG file decompression using jpgd.h
static int Histogram(const char *pSrc_filename, int blocks)
{
	// Features for JPEG image.
	const int req_comps = 3; // request RGB image
	int width = 0;
	int height = 0;
	int actual_comps = 0;
	timer tm;
	
	int *dataR;
	int *dataG;
	int *dataB;

	//---------------------------------------------Load the image from file----------------------------------------------
	uint8 *pImage_data = jpgd::decompress_jpeg_image_from_file(pSrc_filename, &width, &height, &actual_comps, req_comps);
	unsigned char *pixels = pImage_data;

	if (!pImage_data)
	{
		printf("Failed loading JPEG file \"%s\"!\n", pSrc_filename);
		return EXIT_FAILURE;
	}

	//int pos;
	//Print the pixels
	/*for(int i = 0; i < width; i++){
		printf("\n");
		for(int j = 0; j < height; j++){
			pos = ((i * width) + j) * actual_comps;
			printf("[%d, %d, %d]", pixels[pos], pixels[pos + 1], pixels[pos + 2]);
		}
	}
	printf("\n");*/

	printf("Source JPEG file: \"%s\", image resolution: %ix%i, actual comps: ", pSrc_filename, width, height);
	if(actual_comps == 1)
		printf("Gray Scale\n");
	if(actual_comps == 3)
		printf("RGB Scale\n");

	dataR = (int*) malloc (width*height*sizeof(int));
	dataG = (int*) malloc (width*height*sizeof(int));
	dataB = (int*) malloc (width*height*sizeof(int));

	int j = 0;
	for(int i = 0; i < width*height*actual_comps && j < width*height; i += actual_comps){
		dataR[j] = pixels[i];
		dataG[j] = pixels[i + 1];
		dataB[j] = pixels[i + 2];
		j++;
	}
	//--------------------------------------------------------------------------------------------------------------------
	//--------------------------------------------Create the histogram----------------------------------------------------
	histogram_secuencial(dataR, dataG, dataB, width*height);
	histogram_parallel(dataR, dataG, dataB, width*height, blocks);

	//---------------------------------------------------------------------------------------------------------------------

	free(pImage_data);
	free(dataR);
	free(dataG);
	free(dataB);
	return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	if(argc < 3)
	{
		printf("Parameters: [1] Source file (must be JPEG)\n [2] Number of blocks (1 - 1024: Powers of 2)");
		return EXIT_FAILURE;
	}

	const char* pSrc_filename = argv[1]; //filename

	int blocks = atoi( argv[2] );
	bool is_valid = (blocks >= 1 && blocks <= 1024 && ((blocks - 1) & blocks) == 0);
	
    if(!is_valid) {
       blocks = 256;
    }
    
    printf( "Blocks set to %d\n", blocks );
    
	return Histogram(pSrc_filename, blocks);
}
//---------------------------------------------------------------------------------------------
