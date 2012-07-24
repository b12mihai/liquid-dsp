/*
 * Copyright (c) 2007, 2008, 2009, 2010, 2012 Joseph Gaeddert
 * Copyright (c) 2007, 2008, 2009, 2010, 2012 Virginia Polytechnic
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
// Fixed-point matrix (complex)
// 

#include "liquid.internal.h"

#define MATRIX(name)    LIQUID_CONCAT(matrixcq16, name)
#define MATRIX_NAME     "matrixcq16"

// declare fixed-point name-mangling macros
#define LIQUID_FIXED
#define Q(name)         LIQUID_CONCAT(q16,name)
#define CQ(name)        LIQUID_CONCAT(cq16,name)

#define T               cq16_t          // general type
#define TP              q16_t           // primitive type
#define T_COMPLEX       1               // is type complex?

#define MATRIX_PRINT_ELEMENT(X,R,C,r,c)                     \
    printf("%7.2f+j%6.2f ",                                 \
        q16_fixed_to_float(matrix_access(X,R,C,r,c).real),  \
        q16_fixed_to_float(matrix_access(X,R,C,r,c).imag)); \

#include "matrix.base.c"
//#include "matrix.cgsolve.c"
//#include "matrix.chol.c"
//#include "matrix.gramschmidt.c"
//#include "matrix.inv.c"
//#include "matrix.linsolve.c"
//#include "matrix.ludecomp.c"
//#include "matrix.qrdecomp.c"
//#include "matrix.math.c"
