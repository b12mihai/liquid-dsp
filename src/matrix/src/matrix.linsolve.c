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
// Solve linear system of equations
//

#include <string.h>

#include "liquid.internal.h"

// solve linear system of n equations: Ax = b
//  _A      :   system matrix [size: _n x _n]
//  _n      :   system size
//  _b      :   equality vector [size: _n x 1]
//  _x      :   solution vector [size: _n x 1]
//  _opts   :   options (ignored for now)
void MATRIX(_linsolve)(T *          _A,
                       unsigned int _n,
                       T *          _b,
                       T *          _x,
                       void *       _opts,
                       int Mode)
{
	if( Mode == 0 )
	{
		T A_inv[_n*_n];
		clock_t start, stop;
		double t = 0.0;
		memmove(A_inv, _A, _n*_n*sizeof(T));
		start = clock();
		MATRIX(_inv)(A_inv,_n,_n);
		stop = clock();
		t = (double) (stop-start)/CLOCKS_PER_SEC;
		MATRIX(_mul)(A_inv, _n, _n,
					 _b,    _n,  1,
					 _x,    _n,  1);
	}
	else
	{
		unsigned int r;
		unsigned int c;
	
		// compute augmented matrix M [size: _n x _n+1] as
		// T:
		//  A11 A12 A13 ... A1n b1
		//  A21 A22 A23 ... A2n b2
		//  A31 A32 A33 ... A3n b3
		//  ...
		//  An1 An2 An3 ... Ann bn
		T M[_n*_n + _n];    // allocate array
		unsigned int m=0;   // output matrix index counter
		unsigned int a=0;   // input matrix index counter
		for (r=0; r<_n; r++) {
			for (c=0; c<_n; c++)
				M[m++] = _A[a++];
	
			M[m++] = _b[r];
		}
		// run Gauss-Jordan elimination on M
		// T:
		//  1   0   0   ... 0   x1
		//  0   1   0   ... 0   x2
		//  0   0   1   ... 0   x3
		//  ...
		//  0   0   0   ... 1   xn
		MATRIX(_gjelim)(M, _n, _n+1);
	
		// copy result from right-most column of M
		for (r=0; r<_n; r++)
			_x[r] = M[(_n+1)*(r+1)-1];
	}
}


// solve linear system of n equations: Ax = b
//  _A      :   system matrix [size: _n x _n]
//  _n      :   system size
//  _b      :   equality vector [size: _n x 1]
//  _x      :   solution vector [size: _n x 1]
//  _opts   :   options (ignored for now)
void MATRIX(_linsolve_par )(T *          _A,
                       unsigned int _n,
                       T *          _b,
                       T *          _x,
                       void *       _opts,
                       int *counts,
                       int *offsets,
                       int Mode
                       )
{
	if(Mode == 0 )
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		T A_inv[_n*_n];
		clock_t start, stop;
		double t = 0.0;
		memmove(A_inv, _A, _n*_n*sizeof(T));
		MATRIX(_inv_par)(A_inv,_n,_n,counts,offsets);
		MPI_Bcast(_b, _n ,  MPI_FLOAT , 0 , MPI_COMM_WORLD  );
		MATRIX(_mul)(A_inv, _n, _n,
					 _b,    _n,  1,
					 _x,    _n,  1);
	}
	else
	{
		unsigned int r;
		unsigned int c;
	
		// compute augmented matrix M [size: _n x _n+1] as
		// T:
		//  A11 A12 A13 ... A1n b1
		//  A21 A22 A23 ... A2n b2
		//  A31 A32 A33 ... A3n b3
		//  ...
		//  An1 An2 An3 ... Ann bn
		// allocate array
		T M[_n*_n + _n];
		
		// output matrix index counter
		unsigned int m=0;
		// input matrix index counter
		unsigned int a=0;
		/*
		 * in the serial version this implementation copies all the elements
		 * from A into M and then concatenates _b to it
		 */
		for (r=0; r<_n; r++) 
		{
			for ( c = 0; c < _n; c++ )
				M[m++] = _A[a++];
	
			M[m++] = _b[r];
		}
		// run Gauss-Jordan elimination on M
		// T:
		//  1   0   0   ... 0   x1
		//  0   1   0   ... 0   x2
		//  0   0   1   ... 0   x3
		//  ...
		//  0   0   0   ... 1   xn
		MATRIX(_gjelim_par)(M, _n, _n+1);
	
		// copy result from right-most column of M
		for (r=0; r<_n; r++)
			_x[r] = M[(_n+1)*(r+1)-1];
	}
}

