// 
// cgsolve_example.c
//
// Solve linear system of equations A*x = b using the conjugate-
// gradient method where A is a symmetric positive-definite matrix.
// Compare speed to matrixf_linsolve() for same system.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include "liquid.h"
#include "omp.h"
#define PROBLEM_SIZE		1000
#define COMPUTE_DSYMV 		1
#define VERBOSITY			0
int *counts;
int *offsets;
int world, rank;
int *littleCounts;
int *littleOffsets;
int mySize;
unsigned int length;
float* compute_dsymv(unsigned int n, float alpha, float beta, float *A, float *x, float *y)
{
	if(!counts || !offsets ){
		fprintf(stderr,"You passed null parameters in function compute_dsymv\n");
		return NULL;
	}
	 
	float res[mySize];
	float *bigRes;
	float mul[mySize * n ];
	float littleY[mySize*n];
	bigRes = calloc(n, sizeof(float));
	unsigned int i, j;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			matrix_access(A, n, n, i, j) = alpha * matrix_access(A, n, n, i, j);
		}
		y[i] = beta * y[i];
	}
	MPI_Scatterv( A , counts, offsets, MPI_FLOAT , mul , counts[rank], MPI_FLOAT ,
	    				0, MPI_COMM_WORLD);
	/*
	 * multiplying little matrixes
	 */
	matrixf_mul( mul, mySize, n,
                x, n, 1,
                res, mySize, 1);
	
	MPI_Gatherv( res , littleCounts[rank],  MPI_FLOAT , bigRes , littleCounts , 
			littleOffsets , MPI_FLOAT , 0 , MPI_COMM_WORLD);
	
	MPI_Scatterv( y , littleCounts, littleOffsets, MPI_FLOAT , littleY , 
			littleCounts[rank], MPI_FLOAT , 0, MPI_COMM_WORLD);
		
	
	matrixf_add( res , littleY , res , mySize , 1);
	
	MPI_Gatherv( res , littleCounts[rank],  MPI_FLOAT , bigRes , littleCounts , 
			littleOffsets , MPI_FLOAT , 0 , MPI_COMM_WORLD);
	
//	if(rank==0)
//		matrixf_print(bigRes, n,1);
	
	return bigRes;
}
void createParamArrays()
{
	counts = calloc(world, sizeof(int));
	offsets = calloc(world, sizeof(int));
	littleCounts = calloc(world, sizeof(int));
	littleOffsets = calloc(world, sizeof(int));
	unsigned int i;
	for(i=0;i<world;i++)
	{
		counts[i] = length*length/world;
		offsets[i] = i*counts[i];
		littleCounts[i] = length/world;
		littleOffsets[i] = i*littleCounts[i];
	}
}

int main(int argc,char **argv) 
{
	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&world);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);	    
	/*
	 * initialize MPI world
	 */
	
	clock_t start, stop;
	double t = 0.0;
	


    // options
    length = PROBLEM_SIZE;
    if( argc > 1 )
	{
		length = atoi(argv[1]);
		if(!rank)
			printf("\nYou entered a value for problem size %d\n\n",length);    	
		
	}
    unsigned int i,j;
    
    
    // allocate memory for arrays
	float A[length*length];
	float b[length];
	float x[length];
	float x_hat[length];
	float x_prim_hat[length];
	float x_prim_prim_hat[length];
	float L[length*length];
	/*
	 * this is the local matrix of each MPI Process
	 */
	/*
	 * MPI specific variables
	 */
	float mul[mySize*length] ;
	MPI_Status   status;
	float _b[mySize];
	    

    // generate symmetric positive-definite matrix by first generating
    // lower triangular matrix L and computing A = L*L'
    
	createParamArrays();

	
	for ( i=0 ; i < length ; i++) {
		for (j=0; j < length ; j++) {
	#if 0
			// sparse matrix
			if (j > i)              matrix_access(L,length,length,i,j) = 0.0;
			else if (j == i)        matrix_access(L,length,length,i,j) = randnf();
			else if ((rand()%4)==0) matrix_access(L,length,length,i,j) = randnf();
			else                    matrix_access(L,length,length,i,j) = 0.0;
	#else
			// full matrix
			matrix_access(L,length,length,i,j) = (j < i) ? 0.0 : randnf();
	#endif
		}
	}
	

	start = clock();
	if( rank == 0 )
	{
		matrixf_mul_transpose_omp( L , length , length , A );
	}
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	if(!rank)
		printf("1. Mul transpose  : %f \n\n", t );
	MPI_Bcast( A , length*length, MPI_FLOAT , 0 , MPI_COMM_WORLD );
	

	MPI_Bcast( A , length*length, MPI_FLOAT , 0 , MPI_COMM_WORLD );
	
	for (i = 0 ; i < length ; i++)
		x[i] = randnf();
	
	start = clock();
	mySize = length/world;
	int quantity = length*length/world;
	if( rank == 0 )
    {
    	for(i=1;i<world;i++)
    	{
    		MPI_Send( A + i* quantity , quantity , MPI_FLOAT, i, 0 ,MPI_COMM_WORLD);
    	}
    	for(i=0;i<mySize;i++)
    	{
    		for(j=0;j<length;j++)
    		{
    			matrix_access(mul,mySize,length,i,j) = matrix_access(A,length,length,i,j);
    		}
    	}
    		
    }
    else 
    {
    	MPI_Recv(mul , quantity , MPI_FLOAT, 0, MPI_ANY_TAG ,MPI_COMM_WORLD,&status);
    	
    }
	
	matrixf_mul(mul, mySize , length ,
                    x, length , 1,
                    _b, mySize , 1);

    MPI_Gatherv( _b , littleCounts[rank],  MPI_FLOAT , b , littleCounts , 
    		littleOffsets , MPI_FLOAT , 0 , MPI_COMM_WORLD);
    if(rank == 0 )
    {
		stop = clock();
		t = (double) (stop-start)/CLOCKS_PER_SEC;
		printf("2. Multiply : %f \n\n", t );
    }
    	
   // solve symmetric positive-definite system of equations
   start = clock();
   
   matrixf_cgsolve_par( A, length , b, x_hat, mySize, counts, offsets,
   	   littleCounts, littleOffsets,	rank, NULL );
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	if(!rank)
		printf("3. Cgsolve Parallel : %f \n\n", t );
	
	/*
	 * before linsolve we have to adjust parameters
	 */
	start = clock();
	matrixf_linsolve_par( A, length , b, x_prim_hat, NULL,counts,offsets,0);
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	if(!rank)
		printf("4. Linsolve Parallel ver 0 :  %f \n\n", t );
    
	start = clock();
	matrixf_linsolve_par( A, length , b, x_prim_prim_hat, NULL,counts,offsets,1);
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	if( !rank )
		printf("5. Linsolve Parallel ver 1 :  %f \n\n", t );
	
    // compute error norm
    float e = 0.0;
    float e_ = 0.0;
    float e__ = 0.0;

    for (i=0; i < length ; i++)
    {
        e += (x[i] - x_hat[i])*(x[i] - x_hat[i]);
    	e_ += (x[i] - x_prim_hat[i])*(x[i] - x_prim_hat[i]);
    	e__ += (x[i] - x_prim_prim_hat[i])*(x[i] - x_prim_prim_hat[i]);
    }

    e = sqrt(e);
    e_ = sqrt(e_);
    e__ = sqrt(e__);
    if( !rank )
    	printf("error norms: %12.4e , %12.4e , %12.4e,  for rank %d \n", e, e_, e__ , rank );
    
    
#ifdef COMPUTE_DSYMV
    //TODO - alpha and beta could be random
    float alpha = 3.0, beta = 2.0;
    float *res;
    start = clock();
    res = compute_dsymv( length , alpha, beta, A, x, b);
    t = (double) (stop-start)/CLOCKS_PER_SEC;
    if(!rank)
    {
		printf("Run time for compute_dsymv: %f\n", t);
		printf("done.\n");
    }
#endif
    
    // print results
#if VERBOSITY
    printf("A:\n");             matrixf_print(A,     n, n);
    printf("b:\n");             matrixf_print(b,     n, 1);
    printf("x (original):\n");  matrixf_print(x,     n, 1);
    printf("x (estimate via cgsolve):\n");  matrixf_print(x_hat, n, 1);
    printf("x (estimate via linsolve):\n");  matrixf_print(x_prim_hat, n, 1);
    printf("res (dsymv computed result):\n"); matrixf_print(res, n, 1);
#endif
    
    MPI_Finalize();
    return 0;
}

