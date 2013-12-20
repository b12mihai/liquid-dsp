// 
// polyfit_lagrange_example.c
//
// Test exact polynomial fit to sample data using Lagrange
// interpolating polynomials.
// SEE ALSO: polyfit_example.c
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include "liquid.h"
#include <omp.h>

#define OUTPUT_FILENAME "mpi_polyfit_lagrange_example.m"

int size, myid;

#define N_ELEM_PER_PROC		1000

int main(int argc, char **argv)
{

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    float xmin = -1.1f;
    float xmax =  1.1f;
    float xtest = xmin;
    float ytest;
	float *w;
	float *x;
	float *y;
	float *p;
	unsigned int i;
	FILE * fid = fopen(OUTPUT_FILENAME, "w");
    float *sub_x, *sub_y, *sub_p, *sub_w;

    unsigned int n = N_ELEM_PER_PROC * size;      // number of samples
    unsigned int num_steps = 16*n;
    float dx = (xmax-xmin)/(num_steps-1);

	// initialize data vectors
	x = calloc(n, sizeof(float));
	assert(x != NULL);
	y = calloc(n, sizeof(float));
	assert(y != NULL);
	p = calloc(n, sizeof(float));
	assert(p != NULL);
	w = calloc(n, sizeof(float));
	assert (w != NULL);

    if(myid == 0) {

		fprintf(fid,"#!/usr/bin/octave -qf\n");
		fprintf(fid,"%% %s : auto-generated file\n\n", OUTPUT_FILENAME);
		fprintf(fid,"clear all;\nclose all;\n\n");
    }

    sub_x = calloc(N_ELEM_PER_PROC, sizeof(float));
    sub_y = calloc(N_ELEM_PER_PROC, sizeof(float));
    sub_p = calloc(N_ELEM_PER_PROC, sizeof(float));
    sub_w = calloc(N_ELEM_PER_PROC, sizeof(float));
    assert(sub_x != NULL);
    assert(sub_y != NULL);
    assert(sub_p != NULL);
    assert(sub_w != NULL);

    if(myid == 0) {
		for (i=0; i<n; i++) {
			// compute Chebyshev points of the second kind
			x[i] = cosf(M_PI*(float)(i)/(float)(n-1));
			// random samples
			y[i] = 0.2f*randnf();
			//printf("x : %12.8f, y : %12.8f\n", x[i], y[i]);
			fprintf(fid,"x(%3u) = %12.4e; y(%3u) = %12.4e;\n", i+1, x[i], i+1, y[i]);
		}
    }

    MPI_Scatter(x, N_ELEM_PER_PROC, MPI_FLOAT, sub_x, N_ELEM_PER_PROC, MPI_FLOAT, 0, MPI_COMM_WORLD);
    polyf_fit_lagrange(sub_x,y,N_ELEM_PER_PROC,sub_p);
    polyf_fit_lagrange_barycentric(sub_x,N_ELEM_PER_PROC,sub_w);
    MPI_Gather(&sub_p, 1, MPI_FLOAT, p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sub_w, 1, MPI_FLOAT, w, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(myid == 0) {

		// compute Lagrange interpolation weights
		//polyf_fit_lagrange(x,y,n,p);
		//polyf_fit_lagrange_barycentric(x,n,w);

		// print coefficients
		// NOTE : for Chebyshev points of the second kind, w[i] = (-1)^i * (i==0 || i==n-1 ? 1 : 2)
		//for (i=0; i<n; i++)
		//	printf("  w[%3u] = %12.4e;\n", i, w[i]);

		// evaluate polynomial
#pragma omp parallel for reduction(+:xtest) schedule(static, 100)
		for (i=0; i<num_steps; i++) {
			ytest = polyf_val_lagrange_barycentric(x,y,w,xtest,n);
			xtest += dx;
//#pragma omp critical -- introduces overhead, give up printing results, focus on function timings
//			{
//			fprintf(fid,"xtest(%3u) = %12.4e; ytest(%3u) = %12.4e;\n", i+1, xtest, i+1, ytest);
//			}
		}

		// plot results
		fprintf(fid,"plot(x,y,'s',xtest,ytest,'-');\n");
		fprintf(fid,"xlabel('x');\n");
		fprintf(fid,"ylabel('y, p^{(%u)}(x)');\n", n);
		fprintf(fid,"legend('data','poly-fit (barycentric)',0);\n");
		fprintf(fid,"grid on;\n");
		fprintf(fid,"axis([-1.1 1.1 1.5*min(y) 1.5*max(y)]);\n");
		fprintf(fid,"pause()\n");

		fclose(fid);
		printf("results written to %s\n", OUTPUT_FILENAME);

		printf("done.\n");
    }

    MPI_Finalize();

    free(x); free(y); free(p); free(w);
    free(sub_x); free(sub_y); free(sub_p); free(sub_w);

    return 0;
}

