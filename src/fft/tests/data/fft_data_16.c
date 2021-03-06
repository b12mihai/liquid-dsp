/*
 * Copyright (c) 2012 Joseph Gaeddert
 * Copyright (c) 2012 Virginia Polytechnic Institute & State University
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
// autotest fft data for 16-point transform
//

#include <complex.h>

float complex fft_test_x16[] = {
   -1.772146047027 +   0.295934658602*_Complex_I,
   -1.433777343858 +  -0.874041962217*_Complex_I,
   -0.388629405392 +   0.611030474954*_Complex_I,
   -0.492539890742 +   1.007726724574*_Complex_I,
    0.494699992643 +  -1.725668238103*_Complex_I,
    0.572982289851 +   0.061642401846*_Complex_I,
   -0.574974496567 +   0.909843544187*_Complex_I,
    0.733687565510 +   0.447433079732*_Complex_I,
    0.308242485351 +  -1.532252262483*_Complex_I,
    1.207949830231 +  -0.953543898451*_Complex_I,
    0.640048909719 +  -1.022371047059*_Complex_I,
   -0.241879356643 +  -0.462432765300*_Complex_I,
   -0.435900183311 +   0.856847254979*_Complex_I,
    0.577243720893 +   0.220786650383*_Complex_I,
    1.263302572543 +   1.444493924498*_Complex_I,
    1.911070541506 +  -1.906912076526*_Complex_I};

float complex fft_test_y16[] = {
    2.369381184706 +  -2.621483536381*_Complex_I,
   -2.618054253504 +   4.676728363894*_Complex_I,
   -2.946376269367 +   0.522273546089*_Complex_I,
    2.120729100677 +   4.308004588255*_Complex_I,
   -2.975823103566 +  -3.062195121072*_Complex_I,
   -3.932696300262 +   3.143967424649*_Complex_I,
   -0.247865075327 +   5.467645213372*_Complex_I,
    3.887448941709 +   5.439788616063*_Complex_I,
   -3.300093528791 +   2.297200155534*_Complex_I,
   -4.008939493163 +   4.239020717646*_Complex_I,
   -5.630386554230 +  -0.383449644305*_Complex_I,
   -0.707551950987 +   2.197968388714*_Complex_I,
   -1.713879561730 +  -5.034075846097*_Complex_I,
   -8.091926054909 +  -8.469369525668*_Complex_I,
    2.733814414891 +  -7.076455598181*_Complex_I,
   -3.292118248586 +  -0.910613204873*_Complex_I};

