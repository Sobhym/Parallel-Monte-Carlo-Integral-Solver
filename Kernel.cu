#include <cuda.h>	//required for CUDA
#include <curand_kernel.h>
#include <time.h>
#include <limits.h>

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define MAX_N_TERMS 10

__global__ void MC_Integratev1(float* degrees,int dimension,int n_terms,float* I_val,curandState *states, long int seed,int thread_max_iterations)
{
	//Get the Global ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	float x;
	float I = 0.0;
	float f[MAX_N_TERMS];
	//float* f =new float[n_terms];

	//Initialize the random number generator
	curand_init(seed, id, 0, &states[id]);

	for (int iter_count=0;iter_count< thread_max_iterations;iter_count++)
	{
		//Initialize f with the coefficients
		for (int term_i=0;term_i<n_terms;term_i++)
		{
			f[term_i]=degrees[(2+term_i)*dimension];
		}

		for (int d=1;d<dimension;d++)
		{
			//Generate a random number in the range of the limits of this dimension
			x = curand_uniform (&states[id]);    //x between 0 and 1
			//Generate dimension sample based on the limits of the dimension
			x = x*(degrees[1*dimension+d]-degrees[0*dimension+d])+degrees[0*dimension+d];
			for (int term_i=0;term_i<n_terms;term_i++)
			 {
			  	//Multiply f of this term by x^(power of this dimension in this term)
		  		f[term_i]*=pow(x,degrees[(2+term_i)*dimension+d]);
		 	 }
		}
		//Add the evaluation to the private summation
		for (int term_i=0;term_i<n_terms;term_i++)
		{
			I+=f[term_i];
		}
	}
	//Add the private summation to the global summation
	atomicAdd(I_val,I);

}
__global__ void MC_Integratev2(float* degrees_g,int dimension,int n_terms,float* I_val, long int seed,int thread_max_iterations)
{
	//Get the global and local ids
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int lid=threadIdx.x;
	float x;
	float I = 0.0;
	float f[MAX_N_TERMS];
	//float* f =new float[n_terms];

	//Dynamically allocate shared memory for 'degrees' and 'I_shared'
	extern __shared__ float shared_mem[];
	float* I_shared = shared_mem;
	I_shared[0]=0;
	float* degrees = &shared_mem[1];

	//Initialize the local copy of 'degrees' for the shared copy
	if(lid<(2+n_terms)*dimension)
	{
		//copy one element of degrees
		degrees[lid]=degrees_g[lid];
	}

	// Create a state in private memory
	curandState state;
	//Initialize the random number generator
	curand_init(seed,id,0,&state);

	//Synchronize all threads to assure that 'degrees' is initialized
	__syncthreads();

	for (int iter_count=0;iter_count< thread_max_iterations;iter_count++)
	{
		//Initialize f with the coefficients
		for (int term_i=0;term_i<n_terms;term_i++)
		{
			f[term_i]=degrees[(2+term_i)*dimension];
		}

		for (int d=1;d<dimension;d++)
		{
			//Generate a random number in the range of the limits of this dimension
			x = curand_uniform (&state);    //x between 0 and 1
			//Generate dimension sample based on the limits of the dimension
			x = x*(degrees[1*dimension+d]-degrees[0*dimension+d])+degrees[0*dimension+d];
			for (int term_i=0;term_i<n_terms;term_i++)
			 {
				//Multiply f of this term by x^(power of this dimension in this term)
		  		f[term_i]*=pow(x,degrees[(2+term_i)*dimension+d]);
		 	}

		}
		//Add the evaluation to the private summation
		for (int term_i=0;term_i<n_terms;term_i++)
		{
			I+=f[term_i];
		}
	}
	//Add the private summation to the shared summation
	atomicAdd(I_shared,I);
	//Synchronize all the threads to assure they all added their private summations to the shared summation
	__syncthreads();
	//Thread 0 in the block add the shared summation to the global summation
	if(lid==0)
	{
		atomicAdd(I_val,*I_shared);
	}


}


int main(int argc, char** argv)
{

	//----------------------------------
    // Parse Command Line
    //----------------------------------

	if (argc < 8)
	{
		std::cerr << "Required Command-Line Arguments Are:\n";
		std::cerr << "Text file name\n";
		std::cerr << "Method (1 or 2)\n";
		std::cerr << "Dimension \n";
		std::cerr << "Number of Blocks \n";
		std::cerr << "Number of Threads per Block \n";
		std::cerr << "Number of iterations in a thread \n";
		std::cerr << "Validation (1 to validate, 2 validate and show polynomial) \n";
		return -1;
	}


	string filename=argv[1];
    int Method = atol(argv[2]);
    int dimension = atol(argv[3]);
	long long int N_blocks = atol(argv[4]);
	int N_threads = atol(argv[5]);
	int thread_max_iterations=atol(argv[6]);
	int Validate=atol(argv[7]);

	long int max_evaluations = N_blocks*N_threads*thread_max_iterations;

	//----------------------------------
	// Read The file into an array (degrees)
	//----------------------------------

	//Each line in the file represent a term in the polynomial where the first number is the coefficient
	//and the following numbers are the powers of the variables of each dimension in order
	//Line 0: Lower limits for each dimension
	//Line 1: Upper limits for each dimension
	//Accordingly the first element (coefficient) in the first two lines are ignored


	//Add one to the dimension as the first element in every line is the coefficient (first dimension is 1)
	dimension++;

	string line,number;
  	ifstream myfile (filename.c_str());
	float temp=0;
	std::vector<float> degrees (0);

	int line_c=0;
  	if (myfile.is_open())
  	{
		while ( getline (myfile,line) )
    	{
      		std::stringstream linestream(line);
			int number_c=0;
			degrees.resize((line_c+1)*dimension);
			while ( getline (linestream,number,' ' ) )
			{
				stringstream ss;
				ss<<number;
				ss>>temp;
				degrees[line_c*dimension+number_c]=temp;
				number_c++;
			}

			assert(number_c==dimension);
			line_c++;
    	}
		//First two lines are the limits and we need at least one term in the polynomial
		assert(line_c>2);
    	myfile.close();
  	}
  	else cout << "Unable to open file";

	//n_terms: Number of terms in the polynomial (first two lines are the limits)
 	int n_terms=line_c-2;

 	if(n_terms>MAX_N_TERMS)
 	{
 		std::cerr<<"The Maximum Number of terms defined in the code is "<<MAX_N_TERMS<<std::endl;
 		return -1;
 	}
	//----------------------------------
	//Display the numbers in the file (same format as the file)
	//----------------------------------

	if(Validate==2)
	{
		std::cout<<"-----------------------------"<<std::endl;
		std::cout<<"Upper Limit for dimensions = ";
		for(int j=1;j<dimension;j++)
		{
			std::cout<<degrees[j]<<" ";
		}
		std::cout<<std::endl;
		std::cout<<"Lower Limit for dimensions = ";
		for(int j=1;j<dimension;j++)
		{
			std::cout<<degrees[dimension+j]<<" ";
		}
		std::cout<<std::endl;
		for (int i=2;i<line_c;i++)
		{
			std::cout<<"Term "<<i-2<<" Coefficient = ";
			for(int j=0;j<dimension;j++)
			{
				if(j==0)
				{
					std::cout<<degrees[i*dimension+j]<<", Powers = ";
				}
				else
				{
					std::cout<<degrees[i*dimension+j]<<" ";
				}
			}
			std::cout<<std::endl;
		}
		std::cout<<"-----------------------------"<<std::endl;
	}

	//----------------------------------
	//Calculate the Analytical solution
	//----------------------------------

    double Ianalytical=0;
	for (int term_i=0;term_i<n_terms;term_i++)
	{
		double a,b,I;
		//Initialize by the coefficient
		I=degrees[(2+term_i)*dimension];
		for (int d=1;d<dimension;d++)
		{
			a= degrees[0*dimension+d];
			b= degrees[1*dimension+d];
        	b=pow(b,degrees[(2+term_i)*dimension+d]+1);
        	a=pow(a,degrees[(2+term_i)*dimension+d]+1);
			I*=(b-a)/(double)(degrees[(2+term_i)*dimension+d]+1);
		}
		Ianalytical+=I;

	}
	std::cout<<"Analytical Solution = "<< Ianalytical <<std::endl;
	std::cout<<"-----------------------------"<<std::endl;

 	//************************//
 	//		PARALLEL RUN  	 //
 	//***********************//

	std::cout<<"Parallel Case using method "<<Method<<": "<<std::endl;
	std::cout<< "Number of blocks = " << N_blocks <<std::endl;
	std::cout <<"Number of threads per block = " << N_threads<<std::endl;
	std::cout<< "Number of Iterations per thread = " << thread_max_iterations << std::endl;
	std::cout<<"Total number of Evaluations = "<<max_evaluations<<std::endl;
	std::cout<<"Dimension = "<<dimension-1<<std::endl;
	std::cout<<"-----------------------------"<<std::endl;
	//---------------------------------------
    //Initial Setup (Check the Block and grid sizes)
    //---------------------------------------

 	//Get Device properties
 	cudaDeviceProp device_properties;
    cudaGetDeviceProperties	(&device_properties, 0);

    if (N_threads>device_properties.maxThreadsDim[0])
    {
    	std::cerr << "Maximum threads for dimension 0 = " << device_properties.maxThreadsDim[0] << std::endl;
    	return -1;
    }
    if(N_blocks>device_properties.maxGridSize[0])
    {
    	std::cerr << "Maximum grid dimension 0 = " << device_properties.maxGridSize[0] << std::endl;
    	return -1;
    }
 	//---------------------------------------
 	// Setup Profiling
 	//---------------------------------------
 	cudaEvent_t start, stop;
 	cudaEventCreate(&start);
 	cudaEventCreate(&stop);
 	cudaEventRecord(start,0);

	//I_val: Final Estimated Value of the Integral
    float I_val=0;

    //Pointers to data on the device
    float *devDegrees;
    float *dev_I_val;
    curandState *devStates;
	//seed the host random number generator to get a random seed for the Curand
    srand(clock());
	//Random seed to be used for Curand
    long int seed = rand();

	cudaError_t err;
    //Allocate memory for A,B,C on device
    //Pass the address of a pointer where the malloc function will write the address of the data in it
    //it have to be casted to a void pointer
	if (Method!=2)
	{err = cudaMalloc( (void **)&devStates, N_blocks*N_threads * sizeof(curandState) );assert(err == cudaSuccess);}
    err = cudaMalloc((void**)&devDegrees,degrees.size()*sizeof(float));assert(err == cudaSuccess);
    err = cudaMalloc((void**)&dev_I_val,sizeof(float));assert(err == cudaSuccess);

    //Copy the data to the device
    // CudaMemcpy(TO_ADDRESS, 	FROM_ADDRESS,	NUMBER_OF_BYTES,	DIRECTION)
    //Where the direction is either cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
    err = cudaMemcpy( devDegrees,&degrees[0],degrees.size()*sizeof(float),cudaMemcpyHostToDevice);assert(err == cudaSuccess);
    err = cudaMemcpy( dev_I_val,&I_val,sizeof(float),cudaMemcpyHostToDevice);assert(err == cudaSuccess);



    //RUN THE KERNEL
    if(Method==1)
	{
		MC_Integratev1<<<N_blocks,N_threads>>>(devDegrees,dimension,n_terms,dev_I_val,devStates,seed,thread_max_iterations);
	}
	else if (Method ==2)
	{
		MC_Integratev2<<<N_blocks,N_threads,(1+(2+n_terms)*dimension)*sizeof(float)>>>(devDegrees,dimension,n_terms,dev_I_val,seed,thread_max_iterations);
	}
	else
	{
		std::cerr<<"Please enter a valid method"<<std::endl;
		cudaFree(devDegrees);
		cudaFree(dev_I_val);
		return -1;
	}
	//Copy the result to the Host
    err =cudaMemcpy(&I_val,dev_I_val,sizeof(float),cudaMemcpyDeviceToHost);assert(err == cudaSuccess);


    //FREE MEMORY ON DEVICE
    cudaFree(devDegrees);
    cudaFree(dev_I_val);
    if (Method!=2)
    {cudaFree(devStates);}


    //Multiply by the Volume
   	float a,b;
    for (int d=1;d<dimension;d++)
    {
    	a= degrees[0*dimension+d];
    	b= degrees[1*dimension+d];
    	I_val*=(b-a);
    }
    //Divide by the total number of evaluations
	I_val/=(float)N_blocks;
    I_val/=(float)N_threads;
    I_val/=(float)thread_max_iterations;

    //---------------------------------------
    // Stop Profiling
    //---------------------------------------

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);   //time in milliseconds
    gpu_time /= 1000.0;

    std::cout<<"GPU Results: "<<std::endl;
    std::cout <<"I = " << I_val << ", GPU time = "<<gpu_time<<std::endl;



	//******************//
    //	 SERIAL RUN	    //
    //*****************//

	if (Validate==1||Validate==2)
	{
		std::cout<<"-----------------------------"<<std::endl;
		std::cout<<"Host Results: "<<std::endl;

	    double t_start_cpu = (double)clock()/(double)CLOCKS_PER_SEC;
		//Set f_0_s to hold the coefficients of the polynomial terms
	 	std::vector<double> f_0_s (n_terms,0);
 		for (int term_i=0;term_i<n_terms;term_i++)
 		{
 			f_0_s[term_i]=degrees[(2+term_i)*dimension];
 		}

		srand(clock());              //seed the random number generator
		long int N = 0;
		double x;
		double I = 0.0;
		double a,b;

		std::vector<double> f (n_terms,0);
		do
		{
			//Initialize f with the coefficients
			f=f_0_s;
			for (int d=1;d<dimension;d++)
			{
			//Generate a random number in the range of the limits of this dimension
			x = (double)rand()/(double)RAND_MAX;    //x between 0 and 1
			//limits
			a= degrees[0*dimension+d];
			b= degrees[1*dimension+d];
			x = x*(b-a) + a;                        //x between a2 and b2
			for (int term_i=0;term_i<n_terms;term_i++)
			{
				//2: first 2 lines are the limits
				f[term_i]*=pow(x,degrees[(2+term_i)*dimension+d]);
			}
			}
			for (int term_i=0;term_i<n_terms;term_i++)
			{
				I+=f[term_i];
			}
			N++;

		}
		while (N <= max_evaluations);

		//Multiply by the Volume
		for (int d=1;d<dimension;d++)
		{
			a= degrees[0*dimension+d];
			b= degrees[1*dimension+d];
			I*=(b-a);
		}

		I/=(double)N;

		double t_stop_cpu = (double)clock()/(double)CLOCKS_PER_SEC;
		double cpu_time=t_stop_cpu-t_start_cpu;
		std::cout <<"I = " << I << ", Host time = "<<cpu_time<<std::endl;
		std::cout<<"Speed up = "<<cpu_time/gpu_time<<std::endl;
	}



}


