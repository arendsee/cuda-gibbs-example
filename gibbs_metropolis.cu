/*
Zebulun Arendsee
March 26, 2013

This program implements a MCMC algorithm for the following hierarchical
model:

y_k     ~ Poisson(n_k * theta_k)
theta_k ~ Gamma(a, b)
a       ~ Unif(0, a0)
b       ~ Unif(0, b0) 

I let a0 and b0 be arbitrarily large.

Arguments:
    1) input file name
        With two space delimited columns holding integer values for
        y and float values for n.
    2) number of trials (1000 by default)

Output: A comma delimited file containing a column for a, b, and each
theta. All output is written to standard out.

Example:

$ head -3 mydata.txt
 4 0.91643
 23 3.23709
 7 0.40103

$ a.out mydata.txt 2500 > output.csv

This code borrows from the nVidia developer zone documentation, 
specifically http://docs.nvidia.com/cuda/curand/index.html#topic_1_2_1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <curand_kernel.h>



#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)



#define PI 3.14159265359f
#define THREADS_PER_BLOCK 64
#define THREADS_PER_BLOCK_ADD 64



__global__ void setup_kernel(curandState *state, unsigned int seed);
__global__ void sample_theta(curandState *state, float *theta, 
                             int *y, float *n, float a, float b, int N);
__global__ void sum_blocks(float *theta, float *flat_sums, 
                           float *log_sums, int N);
__device__ float rgamma(curandState *state, int id, float a, float b);
__host__ float rnorm();
__host__ float rgamma(float a, float b);
__host__ float sample_a(float a, float b, int N, float log_sum);
__host__ float sample_a(float a, float b, int N, float log_sum);
__host__ float sample_b(float a, int N, float flat_sum);
__host__ void printHeader(int N);



int main(int argc, char *argv[])
{
    curandState *devStates;
    float *dev_theta;
    float *dev_fpsum, *dev_lpsum; 
    int *dev_y;
    float *dev_n;
    int trials = 1000; // Default number of trials
    if(argc > 2)
        trials = atoi(argv[2]);


    /*------ Loading Data ------------------------------------------*/

    FILE *fp;
    if(argc > 1){
        fp = fopen(argv[1], "r");
    } else {
        printf("Please provide input filename\n");
        return 1;
    }

    if(fp == NULL){
        printf("Cannot read file \n");
        return 1;
    }

    int N = 0;
    char line[128];
    while( fgets (line, sizeof line, fp) != NULL ) { 
        N++; 
    }

    rewind(fp);

    int y[N];
    float n[N]; 
    for(int i = 0; i < N; i++){
        fscanf(fp, "%d %f", &y[i], &n[i]);    
    }
   
    fclose(fp);
    


    /*------ Memory Allocations ------------------------------------*/

    CUDA_CALL(cudaMalloc((void **)&dev_y, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(dev_y, y, N * sizeof(int), 
              cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **)&dev_n, N * sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_n, n, N * sizeof(float), 
              cudaMemcpyHostToDevice));

    // Allocate memory for the partial flat and log sums
    int nSumBlocks = (N + (THREADS_PER_BLOCK_ADD - 1)) / THREADS_PER_BLOCK_ADD;
    CUDA_CALL(cudaMalloc((void **)&dev_fpsum, nSumBlocks * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&dev_lpsum, nSumBlocks * sizeof(float)));
    float host_fpsum[N];
    float host_lpsum[N];

    /* Allocate space for theta on device and host */
    CUDA_CALL(cudaMalloc((void **)&dev_theta, N * sizeof(float)));
    float host_theta[N];

    /* Allocate space for random states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, N * sizeof(curandState)));
    


    /*------ Setup RNG ---------------------------------------------*/

    int nBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Setup the rng machines (one for each thread)
    setup_kernel<<<nBlocks, THREADS_PER_BLOCK>>>(devStates, time(NULL));


    /*------ MCMC Algorithm ----------------------------------------*/
    // I commented out the printing functions. Writing the output to
    // the harddrive requires much more time than generating the actual
    // samples. The data are transfered back to the CPU's memory, but
    // currently only the most last result is stored (to avoid memory
    // issues).

//    printHeader(N);

    float a = 20; // Set starting value
    float b = 1;  // Set starting value
    for(int i = 0; i < trials; i++){    
        sample_theta<<<nBlocks, THREADS_PER_BLOCK>>>(devStates, dev_theta,
                                                     dev_y, dev_n, a, b, N);


        // Copy device memory to host
        CUDA_CALL(cudaMemcpy(host_theta, dev_theta, N * sizeof(float), 
                  cudaMemcpyDeviceToHost));
/*
        // Printing the output is by far the most time consuming step
        printf("%f, %f", a, b);
        for(int j = 0; j < N; j++){
            printf(", %f", host_theta[j]);
        }
        printf("\n");
*/
        // Sum the flat and log values of theta 
        sum_blocks<<<nSumBlocks, THREADS_PER_BLOCK_ADD>>>(dev_theta, dev_fpsum,
                                                          dev_lpsum, N);
        
        CUDA_CALL(cudaMemcpy(host_fpsum, dev_fpsum, nSumBlocks * sizeof(float), 
                  cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaMemcpy(host_lpsum, dev_lpsum, nSumBlocks * sizeof(float), 
                  cudaMemcpyDeviceToHost));

        // The GPU summed blocks of theta values, now the CPU sums the blocks
        float flat_sum = 0;
        float log_sum = 0; 
        for(int j = 0; j < nSumBlocks; j++){
            flat_sum += host_fpsum[j];
            log_sum += host_lpsum[j];
        }

        // Sample one random value from a's distribution
        a = sample_a(a, b, N, log_sum);

        // And then from b's distribution given the new a
        b = sample_b(a, N, flat_sum);
    }



    /*------ Free Memory -------------------------------------------*/

    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(dev_theta));
    CUDA_CALL(cudaFree(dev_fpsum));
    CUDA_CALL(cudaFree(dev_lpsum));
    CUDA_CALL(cudaFree(dev_y));
    CUDA_CALL(cudaFree(dev_n));

    return EXIT_SUCCESS;
}



/* 
    Initializes GPU random number generators 
*/
__global__ void setup_kernel(curandState *state, unsigned int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}



/*
    Sample each theta from the appropriate gamma distribution
*/
__global__ void sample_theta(curandState *state, 
                             float *theta, int *y, float *n, 
                             float a, float b, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(id < N){
        float hyperA = a + y[id];
        float hyperB = b + n[id];
        theta[id] = rgamma(state, id, hyperA, hyperB);
    }
}



/* 
    Sampling of a and b require the sum and product of all theta 
    values. This function performs parallel summations of 
    flat values and logs of theta for many blocks of length
    THREADS_PER_BLOCK_ADD. The CPU will then sum the block
    sums (see main method);    
*/
__global__ void sum_blocks(float *theta, float *flat_sums, 
                           float *log_sums, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float flats[THREADS_PER_BLOCK_ADD];
    __shared__ float logs[THREADS_PER_BLOCK_ADD];

    flats[threadIdx.x] = (id < N) ? theta[id] : 0;
    logs[threadIdx.x] = (id < N) ? log( theta[id] ) : 0;
    
    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0){
        if(threadIdx.x < i){
            flats[threadIdx.x] += flats[threadIdx.x + i];
            logs[threadIdx.x] += logs[threadIdx.x + i];
        }
        i /= 2;
    }
    
    if(threadIdx.x == 0){
        flat_sums[blockIdx.x] = flats[0];
        log_sums[blockIdx.x] = logs[0];
    }
}



/* 
    Generate a single Gamma distributed random variable by the Marsoglia 
    algorithm (George Marsaglia, Wai Wan Tsang; 2001).

    I chose this algorithm because it has a very high acceptance rate (>96%),
    so this while loop will usually only need to run a few times. Many other 
    algorithms, while perhaps faster on a CPU, have acceptance rates on the 
    order of 50% (very bad in a massively parallel context).
*/
__device__ float rgamma(curandState *state, int id, float a, float b)
{
    float d = a - 1.0 / 3;
    float Y, U, v;
    while(true){
        // Generate a standard normal random variable
        Y = curand_normal(&state[id]);

        v = pow((1 + Y / sqrt(9 * d)), 3);

        // Necessary to avoid taking the log of a negative number later
        if(v <= 0) 
            continue;
        
        // Generate a standard uniform random variable
        U = curand_uniform(&state[id]);

        // Accept proposed Gamma random variable under following condition,
        // otherise repeat the loop
        if(log(U) < 0.5 * pow(Y,2) + d * (1 - v + log(v)) ){
            return d * v / b;
        }
    }
}



/* 
    Box-Muller Transformation: Generate one standard normal variable.

    This algorithm can be more efficiently used by producing two
    random normal variables. However, for the CPU, much faster
    algorithms are possible (e.g. the Ziggurat Algorithm);

    This is actually the algorithm chosen by nVidia to calculate
    normal random variables on the GPU.
*/
__host__ float rnorm()
{
    float U1 = rand() / float(RAND_MAX);
    float U2 = rand() / float(RAND_MAX);
    float V1 = sqrt(-2 * log(U1)) * cos(2 * PI * U2);
    // float V2 = sqrt(-2 * log(U2)) * cos(2 * PI * U1);
    return V1;
}



/*
    See device rgamma function. This is probably not the
    fastest way to generate random gamma variables on a CPU.
*/
__host__ float rgamma(float a, float b)
{
    float d = a - 1.0 / 3;
    float Y, U, v;
    while(true){
        /* Generate a standard normal random variable */
        Y = rnorm();

        v = pow((1 + Y / sqrt(9 * d)), 3);

        /* If v is negative continue, this is necessary to avoid taking the log
            of a negative number in the next step */
        if(v <= 0) 
            continue;
        
        /* Generate a standard uniform random variable */
        U = rand() / float(RAND_MAX);

        /* Accept proposed Gamma random variable under following condition,
            otherise repeat the loop */
        if(log(U) < 0.5 * pow(Y,2) + d * (1 - v + log(v)) ){
            return d * v / b;
        }
    }
}



/*
    Metropolis algorithm for producing random a values. 
    The proposal distribution in normal with a variance that
    is adjusted at each step.
*/
__host__ float sample_a(float a, float b, int N, float log_sum)
{
    static float sigma = 2;

    float proposal = rnorm() * sigma + a;

    if(proposal <= 0) return a;

    float log_acceptance_ratio = (proposal - a) * log_sum +
                                 N * (proposal - a) * log(b) -
                                 N * (lgamma(proposal) - lgamma(a));

    float U = rand() / float(RAND_MAX);

    if(log(U) < log_acceptance_ratio){
        sigma *= 1.1;
        return proposal;
    } else {
        sigma /= 1.1;
        return a;
    }
}



/*
    Returns a random b sample (simply a draw from the appropriate
    gamma distribution)
*/
__host__ float sample_b(float a, int N, float flat_sum)
{
    float hyperA = N * a + 1;
    float hyperB = flat_sum;
    return rgamma(hyperA, hyperB);
}


/*
    prints: "alpha, beta, theta1, theta2, ... "
*/
__host__ void printHeader(int N)
{
    printf("alpha, beta");
    for(int i = 0; i < N; i++){
        printf(", theta%d", i + 1);
    }
    printf("\n");
}
