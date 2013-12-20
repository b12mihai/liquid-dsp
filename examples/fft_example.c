//
// fft_example.c
//
// This example demonstrates the interface to the fast discrete Fourier
// transform (FFT).
// SEE ALSO: mdct_example.c
//           fct_example.c
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include "liquid.h"


// print usage/help message
void usage()
{
    printf("fft_example [options]\n");
    printf("  h     : print help\n");
    printf("  v/q   : verbose/quiet\n");
    printf("  n     : fft size, default: 16\n");
}

int main(int argc, char*argv[])
{
    // options
    unsigned int nfft = 16; // transform size
    int method = 0;         // fft method (ignored)
    int verbose = 0;        // verbose output?
    fftplan pf, pr;
    int dopt; float rmse = 0.0f;
    float complex d;
    unsigned int i;

    while ((dopt = getopt(argc,argv,"hvqn:")) != EOF) {
        switch (dopt) {
        case 'h': usage();              return 0;
        case 'v': verbose = 1;          break;
        case 'q': verbose = 0;          break;
        case 'n': nfft = atoi(optarg);  break;
        default:
            exit(1);
        }
    }

    // allocate memory arrays
    float complex * x = (float complex*) malloc(nfft*sizeof(float complex));
    float complex * y = (float complex*) malloc(nfft*sizeof(float complex));
    float complex * z = (float complex*) malloc(nfft*sizeof(float complex));
    for (i=0; i<nfft; i++)
        x[i] = (float)i - _Complex_I*(float)i;

    // create fft plans
    pf = fft_create_plan(nfft, x, y, LIQUID_FFT_FORWARD,  method);
    pr = fft_create_plan(nfft, y, z, LIQUID_FFT_BACKWARD, method);

    // execute fft plans
    fft_execute(pf);
    fft_execute(pr);

    // normalize inverse
    for (i=0; i<nfft; i++)
        z[i] /= (float) nfft;
    // initialize input
#pragma omp parallel
{
    // compute RMSE between original and result
#pragma omp for reduction(+:rmse) private(i, d)
    for (i=0; i<nfft; i++) {
        d = x[i] - z[i];
        rmse += crealf(d * conjf(d));
    }
}
    rmse = sqrtf( rmse / (float)nfft );
    printf("rmse = %12.4e\n", rmse);
    if (verbose) {
        // print results
        printf("original signal, x[n]:\n");
        for (i=0; i<nfft; i++)
            printf("  x[%3u] = %8.4f + j%8.4f\n", i, crealf(x[i]), cimagf(x[i]));
        printf("y[n] = fft( x[n] ):\n");
        for (i=0; i<nfft; i++)
            printf("  y[%3u] = %8.4f + j%8.4f\n", i, crealf(y[i]), cimagf(y[i]));
        printf("z[n] = ifft( y[n] ):\n");
        for (i=0; i<nfft; i++)
            printf("  z[%3u] = %8.4f + j%8.4f\n", i, crealf(z[i]), cimagf(z[i]));
    }
    // destroy fft plans
    fft_destroy_plan(pf);
    fft_destroy_plan(pr);

    // free allocated memory
    free(x);
    free(y);
    free(z);

    printf("done.\n");
    return 0;
}

