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
#include <time.h>
#include "liquid.h"

#define PROBLEM_SIZE         1000
#define COMPUTE_DSYMV        1
#define VERBOSITY            0

float* compute_dsymv(unsigned int n, float alpha, float beta, float *A, float *x, float *y)
{
        float *res;
        res = calloc(n, sizeof(float));
        unsigned int i, j;

        for(i = 0; i < n; i++) {
                for(j = 0; j < n; j++) {
                        matrix_access(A, n, n, i, j) = alpha * matrix_access(A, n, n, i, j);
                }
                y[i] = beta * y[i];
        }

        matrixf_mul(A, n, n,
                x, n, 1,
                res, n, 1);
        matrixf_add(res, y, res, n, 1);
        //matrixf_print(res, n, 1 );
                
        return res;
}

int main(int argc,char **argv) {
    // options
    unsigned int n = PROBLEM_SIZE;
    if( argc > 1 )
	{
		n = atoi(argv[1]);
		printf("\nYou entered a value for problem size as %d\n\n",n);    	
		
	}
    clock_t start, stop;
	double t = 0.0;
	start = clock();
    unsigned int i;

    // allocate memory for arrays
    float A[n*n];
    float b[n];
    float x[n];
    float x_hat[n];
    float x_prim_hat[n];
    float x_prim_prim_hat[n];

    // generate symmetric positive-definite matrix by first generating
    // lower triangular matrix L and computing A = L*L'
    float L[n*n];
    unsigned int j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
#if 0
            // sparse matrix
            if (j > i)              matrix_access(L,n,n,i,j) = 0.0;
            else if (j == i)        matrix_access(L,n,n,i,j) = randnf();
            else if ((rand()%4)==0) matrix_access(L,n,n,i,j) = randnf();
            else                    matrix_access(L,n,n,i,j) = 0.0;
#else
            // full matrix
            matrix_access(L,n,n,i,j) = (j < i) ? 0.0 : randnf();
#endif
        }
    }
    
    
	
    start = clock();
	matrixf_mul_transpose(L , n , n, A );
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	printf("1. Mul transpose : %f\n\n", t);
	
    // generate random solution
    for (i=0; i<n; i++)
        x[i] = randnf();

    start = clock();
    // compute b
    matrixf_mul(A, n, n,
                x, n, 1,
                b, n, 1);

    stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	printf("2. Multiply : %f\n\n", t);
	
	start = clock();
    // solve symmetric positive-definite system of equations
    matrixf_cgsolve(A, n, b, x_hat, NULL);
    stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	printf("3. Cgsolve Serial : %f\n\n", t);
    
	
    
    start = clock();
    matrixf_linsolve(A, n, b, x_prim_hat, NULL,0);
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	printf("4. Linsolve serial :  ver 0 : %f\n\n", t);
	
	start = clock();
	matrixf_linsolve(A, n, b, x_prim_prim_hat, NULL,1);
	stop = clock();
	t = (double) (stop-start)/CLOCKS_PER_SEC;
	printf("5. Linsolve serial :  ver 1 : %f\n\n", t);
	
    // compute error norm
    float e = 0.0;
    float e_ = 0.0;
    float e__ = 0.0;

    for (i=0; i<n; i++)
    {
        e += (x[i] - x_hat[i])*(x[i] - x_hat[i]);
        e_ += (x[i] - x_prim_hat[i])*(x[i] - x_prim_hat[i]);
        e__ += (x[i] - x_prim_prim_hat[i])*(x[i] - x_prim_prim_hat[i]);
    }

    e = sqrt(e);
    e_ = sqrt(e_);
    e__ = sqrt(e__);
    printf("error norms: %12.4e , %12.4e %12.4e \n", e, e_,e__);
    
#ifdef COMPUTE_DSYMV
    //TODO - alpha and beta could be random
    float alpha = 3.0, beta = 2.0;
    float *res;
    res = compute_dsymv(n, alpha, beta, A, x, b);
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

    printf("done.\n");
    return 0;
}
