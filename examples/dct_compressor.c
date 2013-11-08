//
// dct_compressor.c
//
// Process a TGA image using DCT. Code is partially inspired from http://unix4lyfe.org/dct/
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "liquid.h"
#include "../tga/targa.h"

#define TGA_ASSERT(x) do { tga_result res;  if ((res=x) != TGA_NOERR) { \
	printf("Targa error: %s\n", tga_error(res)); \
	exit(EXIT_FAILURE); } } while(0)

#define pixel(i,x,y) ( (i)->image_data[((y)*( (i)->width ))+(x)] )


void load_tga(tga_image *tga, const char *fn)
{
	TGA_ASSERT( tga_read(tga, fn) );

	printf("Loaded %dx%dx%dbpp targa (\"%s\").\n",
		tga->width, tga->height, tga->pixel_depth, fn);

	if (!tga_is_mono(tga)) TGA_ASSERT( tga_desaturate_rec_601_1(tga) );
	if (!tga_is_top_to_bottom(tga)) TGA_ASSERT( tga_flip_vert(tga) );
	if (tga_is_right_to_left(tga)) TGA_ASSERT( tga_flip_horiz(tga) );

	//Width and height must be multiplies of 8
	ASSERT ((tga->width % 8 == 0) && (tga->height % 8 == 0));
}


int main(int argc, char **argv)
{

	if(argc < 2) {
		printf("USAGE: dct_compressor PATH_TO_TGA_IMAGE \n");
		ASSERT(argc == 2);
	}

	int type = LIQUID_FFT_REDFT01; // DCT-III
	int flags=0; // FFT flags (typically ignored)


	tga_image tga;
	load_tga(&tga, argv[1]);

	int k = 0;
	int l = (tga.height / 8) * (tga.width / 8);
	int i, j;
	int n = MAXIMUM(tga.height, tga.width);

	float * x = malloc(n * n * sizeof(float ));
	float * y = malloc(n * n * sizeof(float ));

	fftplan q = fft_create_plan_r2r_1d(n * n, x, y, type, flags);

	//initialise input
	for(j = 0; j < tga.height; j++) {
		for(i = 0; i < tga.width; i++) {
			x[j*n + i] = pixel(&tga, i, j);
			y[j*n + i] = 0;
		}
	}

	//Asta face toata magia. Ar trebui sa apeleze functia conform type definit mai sus
	fft_execute(q);

	for(j = 0; j < tga.height; j++) {
		for(i = 0; i < tga.width; i++) {
			pixel(&tga, i, j) = y[j*n + i];
			printf("%f ", y[j*n + i]);
		}
	}

	printf("%d %d %d %d\n", k, l, tga.height, tga.width);
	TGA_ASSERT( tga_write_mono("out.tga", tga.image_data, tga.width, tga.height) );
	//Bunul simt
	fft_destroy_plan(q);
	free(x);
	free(y);

	return 0;
}
