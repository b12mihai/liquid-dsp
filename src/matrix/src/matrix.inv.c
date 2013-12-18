/*
 * Copyright (c) 2007, 2008, 2009, 2010 Joseph Gaeddert
 * Copyright (c) 2007, 2008, 2009, 2010 Virginia Polytechnic
 *                                      Institute & State University
 *
 * This file is part of liquid.
 *
 * liquid is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * liquid is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with liquid.  If not, see <http://www.gnu.org/licenses/>.
 */

//
// Matrix inverse method definitions
//

#include "liquid.internal.h"

void MATRIX(_inv)(T * _X, unsigned int _XR, unsigned int _XC)
{
    // ensure lengths are valid
    if (_XR != _XC ) {
        fprintf(stderr, "error: matrix_inv(), invalid dimensions\n");
        exit(1);
    }
    clock_t start,stop;
    	double  t = 0.0f;
    	
    // X:
    //  x11 x12 ... x1n
    //  x21 x22 ... x2n
    //  ...
    //  xn1 xn2 ... xnn

    // allocate temporary memory
    T x[2*_XR*_XC];
    unsigned int xr = _XR;
    unsigned int xc = _XC*2;

    // x:
    //  x11 x12 ... x1n 1   0   ... 0
    //  x21 x22 ... x2n 0   1   ... 0
    //  ...
    //  xn1 xn2 ... xnn 0   0   ... 1
    unsigned int r,c;
    for (r=0; r<_XR; r++) {
        // copy matrix elements
        for (c=0; c<_XC; c++)
            matrix_access(x,xr,xc,r,c) = matrix_access(_X,_XR,_XC,r,c);

        // append identity matrix
        for (c=0; c<_XC; c++)
            matrix_access(x,xr,xc,r,_XC+c) = (r==c) ? 1 : 0;
    }
    
    // perform Gauss-Jordan elimination on x
    // x:
    //  1   0   ... 0   y11 y12 ... y1n
    //  0   1   ... 0   y21 y22 ... y2n
    //  ...
    //  0   0   ... 1   yn1 yn2 ... ynn
    	start = clock();
    MATRIX(_gjelim)(x,xr,xc);
    // copy result from right half of x
    for (r=0; r<_XR; r++) {
        for (c=0; c<_XC; c++)
            matrix_access(_X,_XR,_XC,r,c) = matrix_access(x,xr,xc,r,_XC+c);
    }
}
// Gauss-Jordan elmination
void MATRIX(_gjelim)(T * _X, unsigned int _XR, unsigned int _XC)
{
    unsigned int r, c;
    // choose pivot rows based on maximum element along column
    float v;
    float v_max=0.;
    unsigned int r_opt=0;
    unsigned int r_hat;
    clock_t start,stop;
    double  t = 0.0f;
    for (r=0; r<_XR; r++) {

		// check values along this column and find the optimal row
		for (r_hat=r; r_hat<_XR; r_hat++) {
			v = cabsf( matrix_access(_X,_XR,_XC,r_hat,r) );
			// swap rows if necessary
			if (v > v_max || r_hat==r) {
				r_opt = r_hat;
				v_max = v;
			}
		}
		// if the maximum is zero, matrix is singular
		if (v_max == 0.0f) {
			fprintf(stderr,"warning: matrix_gjelim(), matrix singular to machine precision\n");
		}
		
		// if row does not match column (e.g. maximum value does not
		// lie on the diagonal) swap the rows
		if (r != r_opt) 
		{
			MATRIX(_swaprows)(_X,_XR,_XC,r,r_opt);
		}

		// pivot on the diagonal element
		MATRIX(_pivot)(_X,_XR,_XC,r,r);
	}
	// scale by diagonal
    T g;
    for (r=0; r<_XR; r++) {
        g = 1 / matrix_access(_X,_XR,_XC,r,r);
        for (c=0; c<_XC; c++)
            matrix_access(_X,_XR,_XC,r,c) *= g;
    }
}
// pivot on element _r, _c
void MATRIX(_pivot)(T * _X, unsigned int _XR, unsigned int _XC, unsigned int _r, unsigned int _c)
{
    T v = matrix_access(_X,_XR,_XC,_r,_c);
    if (v==0) 
    {
        fprintf(stderr, "warning: matrix_pivot(), pivoting on zero matrix_inv 607 \n");
        return;
    }
    unsigned int r,c;

    // pivot using back-substitution
    T g;    // multiplier
    for (r=0; r<_XR; r++) {

        // skip over pivot row
        if (r == _r)
            continue;

        // compute multiplier
        g = matrix_access(_X,_XR,_XC,r,_c) / v;
        // back-substitution
        for (c=0; c<_XC; c++) 
        {
        	matrix_access(_X,_XR,_XC,r,c) = g*matrix_access(_X,_XR,_XC,_r,c) -
                                              matrix_access(_X,_XR,_XC, r,c);
        }
    }
}

void MATRIX(_inv_par)(T * _X, unsigned int _XR, unsigned int _XC , int *counts, int *offsets )
{
    // ensure lengths are valid
    if ( _XR != _XC ) 
    {
        fprintf(stderr, "error: matrix_inv(), invalid dimensions\n");
        exit(1);
    }
    // X:
    //  x11 x12 ... x1n
    //  x21 x22 ... x2n
    //  ...
    //  xn1 xn2 ... xnn

    // x:
    //  x11 x12 ... x1n 1   0   ... 0
    //  x21 x22 ... x2n 0   1   ... 0
    //  ...
    //  xn1 xn2 ... xnn 0   0   ... 1
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
        
    int i,j;
    unsigned int _littleXR = _XR/size;
    T X[2*_XR*_XC];
    // allocate temporary memory
	T x[2*_XR/size*_XC];
	unsigned int xr = _XR / size ;
	unsigned int xc = _XC*2;
	unsigned int r,c;
	clock_t start,stop;
	double  t = 0.0f;
	
    /*
     * allocate the local matrix of each mpi process
     */
    float _littleX[ _XR/size * _XC ];
    MPI_Scatterv( _X , counts, offsets, MPI_FLOAT , _littleX , counts[rank] , MPI_FLOAT ,
             				0, MPI_COMM_WORLD);
    /*
     * the scatterV unfortunately does not scatter also in the 'scatterer' local matrix
     */
    if(rank == 0 )
    {
    	for(i=0;i<_littleXR;i++)
    		for(j=0;j<_XC;j++)
    			matrix_access(_littleX,xr,_XC,i,j) = matrix_access(_X , _XR , _XC , i , j);
    }
    
    for ( r = 0 ; r < xr ; r++ ) 
    {
        // copy matrix elements
        for (c=0; c<_XC; c++)
            matrix_access(x,xr,xc,r,c) = matrix_access(_littleX , xr , _XC , r , c);
        int actualRow = (r+rank*_XR/size);
        // append identity matrix
        for (c = 0; c < _XC; c++)
        {
            matrix_access(x,xr,xc,r,_XC+c ) = ( actualRow   == c ) ? 1 : 0;
        }
    }
    
    /*
     * gauss jordan elimination is only possible on square matrix so it would be quit
     * hard to paralalize there
     */
    
    
    // perform Gauss-Jordan elimination on x
    // x:
    //  1   0   ... 0   y11 y12 ... y1n
    //  0   1   ... 0   y21 y22 ... y2n
    //  ...
    //  0   0   ... 1   yn1 yn2 ... ynn
  
    /*
     * we have to adjust the array for this gather , multiply by 2
     * cause of identity matrix
     */
    int c2[size];
    int of2[size];
    for(i=0;i<size;i++)
    {
    	c2[i] = counts[i] * 2;
    	of2[i] = i*c2[i];
    }
    
    MPI_Gatherv( x , c2[rank],  MPI_FLOAT , X , c2 , of2 
    				, MPI_FLOAT ,0, MPI_COMM_WORLD);
    
    /*
     * only master does the gauss thing
     * we ll see about that
     */
	start = clock();
	MPI_Bcast( X, 2*_XR*_XC , MPI_FLOAT , 0 , MPI_COMM_WORLD );
	
	MATRIX(_gjelim_par)(X,_XR,xc);
	
	
   /*
    * this copying actually does not take long so i did not paralle... it
    */
   if(rank == 0)
	{
		// copy result from right half of x
		for ( r =0 ; r < _XR; r++ ) 
		{
			for (c=0; c<_XC; c++)
			{
				matrix_access(_X,_XR,_XC,r,c) = matrix_access(X,_XR,xc,r,_XC+c);
			}
		}
	
	}
    /*
     * at the end make the other processes aware of the _X
     */
    MPI_Bcast( _X,_XR*_XC, MPI_FLOAT , 0 , MPI_COMM_WORLD );
    
}

// Gauss-Jordan elmination
void MATRIX(_gjelim_par)(T * _X, unsigned int _XR, unsigned int _XC)
{
    unsigned int r, c;
    int rank,size, i;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    // choose pivot rows based on maximum element along column
    float v;
    float v_max=0.;
    float global_max=0.;
    unsigned int r_opt=0;
    unsigned int r_hat;
    
    /*
     * MPI specific
     */
    
    int _xr = _XR/size;
    float _x[_xr*_XC];
    int counts[size],offsets[size];
    
    for(i = 0;i < size;i++)
    {
    	counts[i] = _xr*_XC;
    	offsets[i] = i*counts[i];
    }
    
    MPI_Scatterv( _X , counts, offsets, MPI_FLOAT , _x , counts[rank] , MPI_FLOAT ,
                 				0, MPI_COMM_WORLD);
    
	MPI_Status status;
	MPI_Request request;
	int limSup = (rank + 1)*_xr -1;
	int limInf = rank *_xr;
	int start = 0 ;
	clock_t startcl,stop;
	double  t = 0.0f;
	int aux = rank*_xr;
	
	
	for (r=0; r<_XR; r++) 
	{
    	start = 0;
    	if( r >= limInf && r <= limSup )
    		start = r % _xr;
    	else if( r > limSup )
    	{
    		start  = r;
    	}
    	if(start > limSup)
    		v_max = 0.0f;
    	
    	// check values along this column and find the optimal row
    	
		for ( r_hat = start ; r_hat<_xr; r_hat++ ) 
		{
			v = cabsf( matrix_access(_x,_xr,_XC,r_hat, r ) );
			// swap rows if necessary
			if ( v > v_max || r_hat == start  ) 
			{
				r_opt = r_hat + aux ;
				v_max = v;
			}
		}
		
		MPI_Reduce(&v_max , &global_max, 1 , MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD );
		MPI_Bcast( &global_max , 1, MPI_FLOAT, 0 , MPI_COMM_WORLD );
		
		/*
		 * every thread checks to see if it has the maximum value
		 */
		if( v_max == global_max )
		{
			
			/*
			 * check the r_opt
			 */
			for(i=0;i<size;i++)
			{
				if(i == rank)
					continue;
				MPI_Send( &r_opt , 1, MPI_INT , i ,2, MPI_COMM_WORLD );
			}
			
			if( r_opt != r )
			{
				int dest = r/_xr;
				int row = r % _xr;
				int row2 = r_opt % _xr;
				if( dest == rank )
				{
					MATRIX(_swaprows)(_x,_xr,_XC, row , row2 );
									
				}
				else
				{
					int offset = (r_opt % _xr)*_XC;
					MPI_Send( _x + offset , _XC , MPI_FLOAT , dest , 1 , MPI_COMM_WORLD );
					MPI_Recv( _x + offset , _XC , MPI_FLOAT , MPI_ANY_SOURCE , 8 , MPI_COMM_WORLD, &status );
				}
			}
		}
		if( v_max != global_max )
		{
			MPI_Recv( &r_opt , 1 , MPI_INT , MPI_ANY_SOURCE , 2 , MPI_COMM_WORLD,&status );
		}
		int me = r/_xr;
		int s = r_opt /_xr;
		
		if( me == rank && me != s)
		{
			int offset = r % _xr;
			MPI_Send( _x + offset*_XC , _XC , MPI_FLOAT , r_opt/_xr , 8 , MPI_COMM_WORLD);
			MPI_Recv( _x + offset*_XC , _XC , MPI_FLOAT , MPI_ANY_SOURCE , 1 , MPI_COMM_WORLD, &status );
		}
		
		MATRIX(_pivot_mpi)(_x,_xr,_XC,r,r);
	}
	
	MPI_Gatherv(  _x , counts[rank] , MPI_FLOAT , _X , counts , offsets, MPI_FLOAT ,	0, MPI_COMM_WORLD);
	
	// scale by diagonal
    if(rank == 0 )
    {
		for (r=0; r<_XR; r++) 
		{
			T g = 1 / matrix_access(_X,_XR,_XC,r,r);
			for (c=0; c<_XC; c++)
				matrix_access(_X,_XR,_XC,r,c) *= g;
		}
    }
    
}
// Gauss-Jordan elmination
void MATRIX(_gjelim_par2)(T * _X, unsigned int _XR, unsigned int _XC)
{
    unsigned int r, c;
    int rank,size, i;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    // choose pivot rows based on maximum element along column
    float v;
    float v_max=0.;
    float global_max=0.;
    unsigned int r_opt=0;
    unsigned int r_hat;
    
    /*
     * MPI specific
     */
    
    int _xr = _XR/size;
    float _x[_xr*_XC];
    int counts[size],offsets[size];
    
    for(i = 0;i < size;i++)
    {
    	counts[i] = _xr*_XC;
    	offsets[i] = i*counts[i];
    }
    
    MPI_Scatterv( _X , counts, offsets, MPI_FLOAT , _x , counts[rank] , MPI_FLOAT ,
                 				0, MPI_COMM_WORLD);
    
	MPI_Status status;
	MPI_Request request;
	int limSup = (rank + 1)*_xr -1;
	int limInf = rank *_xr;
	int start = 0 ;
	clock_t startcl,stop;
	double  t = 0.0f;
	int aux = rank*_xr;
	startcl = clock();
	for (r=0; r<_XR; r++) 
	{
    	start = 0;
    	if( r >= limInf && r <= limSup )
    		start = r % _xr;
    	else if( r > limSup )
    	{
    		counts[rank] = 0;
    		start  = r;
    	}
    	if(start > limSup)
    		v_max = 0.0f;
		
    	// check values along this column and find the optimal row
    	
		for ( r_hat = start ; r_hat<_xr; r_hat++ ) 
		{
			v = cabsf( matrix_access(_x,_xr,_XC,r_hat, r ) );
			// swap rows if necessary
			if ( v > v_max || r_hat == start  ) 
			{
				r_opt = r_hat + aux ;
				v_max = v;
			}
		}
		
		MPI_Reduce(&v_max , &global_max, 1 , MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD );
		MPI_Bcast( &global_max , 1, MPI_FLOAT, 0 , MPI_COMM_WORLD );
		
		/*
		 * every thread checks to see if it has the maximum value
		 */
		
		if( rank ==  0 )
		{
			if( v_max != global_max )
			{
				MPI_Recv(&r_opt, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD , &status );
			}
		}
		if(rank !=  0 )
		{
			if( global_max == v_max )
			{
				MPI_Isend(&r_opt, 1, MPI_INT, 0 ,1, MPI_COMM_WORLD , &request );
			}
		}
		
		if( rank == 0 )
		{
			// if the maximum is zero, matrix is singular
			if ( global_max == 0.0f ) 
			{
				fprintf(stderr,"warning: matrix_gjelim(), matrix singular to machine precision\n");
			}
	
			// if row does not match column (e.g. maximum value does not
			// lie on the diagonal) swap the rows
			if (r != r_opt) 
			{
				MATRIX(_swaprows)(_X,_XR,_XC,r,r_opt);
				
			}
	
			// pivot on the diagonal element
			MATRIX(_pivot)(_X,_XR,_XC,r,r);
			/*
			 * after all modification master has to inform slaves
			 */
		}
		stop = clock();
		t = (double) (stop-startcl);
		MPI_Scatterv( _X , counts, offsets, MPI_FLOAT , _x , counts[rank] , MPI_FLOAT ,
														0, MPI_COMM_WORLD);
		
		stop = clock();
		t = (double) (stop-startcl);
	}

	// scale by diagonal
    if( rank == 0 )
    {
		for (r=0; r<_XR; r++) 
		{
			T g = 1 / matrix_access(_X,_XR,_XC,r,r);
			for (c=0; c<_XC; c++)
				matrix_access(_X,_XR,_XC,r,c) *= g;
		}
    }
   
}

// pivot on element _r, _c
void MATRIX(_pivot_mpi)(float * _X, unsigned int _XR, unsigned int _XC, unsigned int _r, unsigned int _c)
{
    float v = 0.0f ;
    int probably_me = _r/_XR;
    int rank ;
    int i,size ; 
    float *aux=calloc(_XC, sizeof(float));
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;
    if( probably_me == rank )
    {
    	v = matrix_access(_X,_XR,_XC,_r%_XR,_c);
    	aux = _X + (_r%_XR*_XC);
    	for(i=0;i<size;i++)
		{
    		/*
    		 * do not also send to myself
    		 */
			if( i == rank )
				continue;
			MPI_Send( &v , 1, MPI_FLOAT, i ,3, MPI_COMM_WORLD );
			MPI_Send( aux , _XC , MPI_FLOAT, i ,4, MPI_COMM_WORLD );
		}
    }
    else
    {
    	MPI_Recv( &v , 1 , MPI_FLOAT , MPI_ANY_SOURCE , 3 , MPI_COMM_WORLD,&status );
    	MPI_Recv( aux , _XC , MPI_FLOAT , MPI_ANY_SOURCE , 4 , MPI_COMM_WORLD,&status );
    }
    
    if (v == 0.0f) 
    {
    	fprintf(stderr, "warning: matrix_pivot(), pivoting on zero matrix.inv 575 \n");
        return;
    }
    unsigned int r,c;

    // pivot using back-substitution
    T g;    // multiplier
    int adaos = _XR*rank;
    for (r=0; r<_XR; r++) {

        // skip over pivot row
        if ( ( adaos + r ) == _r)
            continue;

        // compute multiplier
        g = matrix_access(_X,_XR,_XC,r,_c) / v;

        /*
         * back-substitution
         */
        for (c=0; c<_XC; c++) 
        {
        	matrix_access(_X,_XR,_XC,r,c) = g*aux[c] -  matrix_access(_X,_XR,_XC, r,c);
        }
    }
}



void MATRIX(_swaprows)(T * _X, unsigned int _XR, unsigned int _XC, unsigned int _r1, unsigned int _r2)
{
    unsigned int c;
//#pragma omp for schedule(dynamic,100)
    for (c=0; c<_XC; c++) 
    {
        T v_tmp = matrix_access(_X,_XR,_XC,_r1,c);
        matrix_access(_X,_XR,_XC,_r1,c) = matrix_access(_X,_XR,_XC,_r2,c);
        matrix_access(_X,_XR,_XC,_r2,c) = v_tmp;
    }
}


