//
// linsolve_all_example.c
//
// Solve linear system of equations A*x = b using all methodes provided in liquid
// DSP and some mathematical tricks for benchmarking purposes
//
// General assumptions are: A = n x n matrix, x = n dim array, b = n dim array
// n is given as unsigned int, global constant variable across functions

//Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "liquid.h"

//Defines go here!

#ifndef T
#define T 		float			//benchmarking should be done also float vs. double. Fun would be to try complex :)
#endif

#define RAND_A		100

//TODO: all these should be done from makefile
#define RANDOM_DATA
//#define	 	VERBOSITY
#define 		DEBUG

unsigned int n = 5;

//Used for debugging results/comparing floats
const double eps = 0.01;

void solve_upper(T *U, T *x, T *b)
{
	int i, k;
	T sum = 0;

	x[n - 1] = b[n - 1] / matrix_access(U, n, n, n - 1, n - 1);

	for(i = n - 2; i >= 0; i--) {
		sum = 0;
		for(k = i + 1; k < n; k++) {
			sum += matrix_access(U, n, n, i, k) * x[k];
		}

		x[i] = (b[i] - sum) / matrix_access(U, n, n, i, i);
	}


}

void solve_lower(T *L, T *x, T *b)
{
	int i, k;
	T sum = 0;

	x[0] = b[0] / matrix_access(L, n, n, 0, 0);

	for(i = 1; i < n; i++) {
		sum = 0;
		for(k = 0; k <= i - 1; k++) {
			sum += matrix_access(L, n, n, i, k) * x[k];
		}
		x[i] = (b[i] - sum) / matrix_access(L, n, n, i, i);
	}

}

void solve_lu_system(T *L, T *U, T *x, T *b)
{
	T *y;
	y = calloc(n, sizeof(T));
	solve_lower(L, y, b);
	solve_upper(U, x, y);
	free(y);
}

void print_results(T *A, T *x, T *b)
{
	T *b_hat;
	float err = 0.0;
	b_hat = calloc(n, sizeof(T));
	int i;

#ifdef VERBOSITY
    printf("A:\n");            		matrixf_print(A,     n, n);
    printf("b:\n");             	matrixf_print(b,     n, 1);
    printf("x (solution) : \n"); 	matrixf_print(x,     n, 1);
#endif

    matrixf_mul(A, n, n,
                x, n, 1,
                b_hat, n, 1);
    for (i=0; i<n; i++) {
    	err += (b[i] - b_hat[i])*(b[i] - b_hat[i]);

#if 0
    	if(fabs(b[i] - b_hat[i]) > eps) {
    		printf("Detected high error %f at position %d \n", fabs(b[i] - b_hat[i]), i);
    	}
#endif
    }

    err = sqrt(err);
    printf("error norm: %.6f\n", err);

    free(b_hat);

}

// For LU decomposition of Ax = b, A = LU => two systems Ly = b and Ux = y

int main(int argc, char **argv)
{
	if (argc > 1) {
		n = atoi(argv[1]);
	}

	ASSERT(n != 0);
	printf("~~~~~~~Running Ax = b, A = n*n matrix, x, b n-dim arrays with n = %d \n~~~~~~~", n);

	unsigned int i, j;

	//Linear system Ax = b
	T *A, *x, *b;

	//L/U/P decomposition. We use particular case P = eye(n)
	T *L, *U, *P;

	A = calloc(n * n, sizeof(T));
	L = calloc(n * n, sizeof(T));
	U = calloc(n * n, sizeof(T));
	P = calloc(n * n, sizeof(T));
	x = calloc(n, sizeof(T));
	b = calloc(n, sizeof(T));

	srand(time(NULL));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
//#ifdef RANDOM_DATA
            matrix_access(A, n, n, i, j) = (float)rand()/(float)(RAND_MAX/RAND_A); //in order to be positive definite for Cholesky
            b[i] = (float)rand()/(float)(RAND_MAX/RAND_A);
//#else
            //TODO: import from file, matrixmarket :)
//#endif
        }
    }

    printf("~~~~~~~Running simple linear solving ~~~~~~~\n");
    matrixf_linsolve(A, n, b, x, NULL);

#ifdef DEBUG
    print_results(A, x, b);
#endif

    printf("~~~~~~~Running LU decomposition using Crout method~~~~~~~\n");
    matrixf_ludecomp_crout(A, n, n, L, U, P);
    solve_lu_system(L, U, x, b);

#ifdef VERBOSITY
    printf("L:\n");            		matrixf_print(L,     n, n);
    printf("U:\n");             	matrixf_print(U,     n, n);
#endif


#ifdef DEBUG
    print_results(A, x, b);
#endif

    printf("~~~~~~~Running LU decomposition using Doolittle method~~~~~~~\n");
    matrixf_ludecomp_doolittle(A, n, n, L, U, P);
    solve_lu_system(L, U, x, b);

#ifdef VERBOSITY
    printf("L:\n");            		matrixf_print(L,     n, n);
    printf("U:\n");             	matrixf_print(U,     n, n);
#endif


#ifdef DEBUG
    print_results(A, x, b);
#endif

#if 0
	   /* TODO: issue with not positive definite matrix A at Cholesky...HELP */
		printf("~~~~~~~Running LU decomposition using Cholesky method~~~~~~~\n");
		L = calloc(n * n, sizeof(T));
		U = calloc(n * n, sizeof(T));
		matrixf_chol(A, n, L);
		matrixf_transpose_mul(L, n, n, U);
		solve_lu_system(L, U, x, b);

	#ifdef VERBOSITY
		printf("L:\n");            		matrixf_print(L,     n, n);
		printf("U:\n");             	matrixf_print(U,     n, n);
	#endif

	#ifdef DEBUG
		print_results(A, x, b);
	#endif
#endif

#if 0
	printf("~~~~~~~Running QR decomposition using GramSchmidt method~~~~~~~\n");
	//L = Q, U = R -- recycling ftw
	matrixf_qrdecomp_gramschmidt(A, n, n, L, U);
	matrixf_transpose_mul(L, n, n, P);
	matrixf_mul(P, n, n,
			    b, n, 1,
			    b, n, 1);
	solve_upper(U, x, b);
#ifdef VERBOSITY
    printf("Q:\n");            		matrixf_print(L,     n, n);
    printf("R:\n");             	matrixf_print(U,     n, n);
#endif
#ifdef DEBUG
    print_results(A, x, b);
#endif
#endif

    free(A);
    free(x);
    free(b);
    free(L);
    free(U);
    free(P);

    return 0;
}
