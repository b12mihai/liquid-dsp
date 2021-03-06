/*
 * Copyright (c) 2007, 2009, 2012, 2013 Joseph Gaeddert
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
// fft_composite_benchmark.c : benchmark FFTs of 'composite' length (not
//   prime, not of form 2^m)
//

#include <stdlib.h>
#include <stdio.h>
#include <sys/resource.h>
#include "liquid.h"

#include "src/fft/bench/fft_runbench.h"

// composite numbers
void benchmark_fft_6      LIQUID_FFT_BENCHMARK_API(     6, LIQUID_FFT_FORWARD)
void benchmark_fft_9      LIQUID_FFT_BENCHMARK_API(     9, LIQUID_FFT_FORWARD)
void benchmark_fft_10     LIQUID_FFT_BENCHMARK_API(    10, LIQUID_FFT_FORWARD)
void benchmark_fft_12     LIQUID_FFT_BENCHMARK_API(    12, LIQUID_FFT_FORWARD)
void benchmark_fft_14     LIQUID_FFT_BENCHMARK_API(    14, LIQUID_FFT_FORWARD)
void benchmark_fft_15     LIQUID_FFT_BENCHMARK_API(    15, LIQUID_FFT_FORWARD)
void benchmark_fft_18     LIQUID_FFT_BENCHMARK_API(    18, LIQUID_FFT_FORWARD)
void benchmark_fft_20     LIQUID_FFT_BENCHMARK_API(    20, LIQUID_FFT_FORWARD)
void benchmark_fft_21     LIQUID_FFT_BENCHMARK_API(    21, LIQUID_FFT_FORWARD)
void benchmark_fft_22     LIQUID_FFT_BENCHMARK_API(    22, LIQUID_FFT_FORWARD)
void benchmark_fft_24     LIQUID_FFT_BENCHMARK_API(    24, LIQUID_FFT_FORWARD)
void benchmark_fft_25     LIQUID_FFT_BENCHMARK_API(    25, LIQUID_FFT_FORWARD)
void benchmark_fft_26     LIQUID_FFT_BENCHMARK_API(    26, LIQUID_FFT_FORWARD)
void benchmark_fft_27     LIQUID_FFT_BENCHMARK_API(    27, LIQUID_FFT_FORWARD)
void benchmark_fft_28     LIQUID_FFT_BENCHMARK_API(    28, LIQUID_FFT_FORWARD)
void benchmark_fft_30     LIQUID_FFT_BENCHMARK_API(    30, LIQUID_FFT_FORWARD)
void benchmark_fft_33     LIQUID_FFT_BENCHMARK_API(    33, LIQUID_FFT_FORWARD)
void benchmark_fft_34     LIQUID_FFT_BENCHMARK_API(    34, LIQUID_FFT_FORWARD)
void benchmark_fft_35     LIQUID_FFT_BENCHMARK_API(    35, LIQUID_FFT_FORWARD)
void benchmark_fft_36     LIQUID_FFT_BENCHMARK_API(    36, LIQUID_FFT_FORWARD)
void benchmark_fft_38     LIQUID_FFT_BENCHMARK_API(    38, LIQUID_FFT_FORWARD)
void benchmark_fft_39     LIQUID_FFT_BENCHMARK_API(    39, LIQUID_FFT_FORWARD)
void benchmark_fft_40     LIQUID_FFT_BENCHMARK_API(    40, LIQUID_FFT_FORWARD)
void benchmark_fft_42     LIQUID_FFT_BENCHMARK_API(    42, LIQUID_FFT_FORWARD)
void benchmark_fft_44     LIQUID_FFT_BENCHMARK_API(    44, LIQUID_FFT_FORWARD)
void benchmark_fft_45     LIQUID_FFT_BENCHMARK_API(    45, LIQUID_FFT_FORWARD)
void benchmark_fft_46     LIQUID_FFT_BENCHMARK_API(    46, LIQUID_FFT_FORWARD)
void benchmark_fft_48     LIQUID_FFT_BENCHMARK_API(    48, LIQUID_FFT_FORWARD)
void benchmark_fft_49     LIQUID_FFT_BENCHMARK_API(    49, LIQUID_FFT_FORWARD)
void benchmark_fft_50     LIQUID_FFT_BENCHMARK_API(    50, LIQUID_FFT_FORWARD)
void benchmark_fft_51     LIQUID_FFT_BENCHMARK_API(    51, LIQUID_FFT_FORWARD)
void benchmark_fft_52     LIQUID_FFT_BENCHMARK_API(    52, LIQUID_FFT_FORWARD)
void benchmark_fft_54     LIQUID_FFT_BENCHMARK_API(    54, LIQUID_FFT_FORWARD)
void benchmark_fft_55     LIQUID_FFT_BENCHMARK_API(    55, LIQUID_FFT_FORWARD)
void benchmark_fft_56     LIQUID_FFT_BENCHMARK_API(    56, LIQUID_FFT_FORWARD)
void benchmark_fft_57     LIQUID_FFT_BENCHMARK_API(    57, LIQUID_FFT_FORWARD)
void benchmark_fft_58     LIQUID_FFT_BENCHMARK_API(    58, LIQUID_FFT_FORWARD)
void benchmark_fft_60     LIQUID_FFT_BENCHMARK_API(    60, LIQUID_FFT_FORWARD)
void benchmark_fft_62     LIQUID_FFT_BENCHMARK_API(    62, LIQUID_FFT_FORWARD)
void benchmark_fft_63     LIQUID_FFT_BENCHMARK_API(    63, LIQUID_FFT_FORWARD)
void benchmark_fft_65     LIQUID_FFT_BENCHMARK_API(    65, LIQUID_FFT_FORWARD)
void benchmark_fft_66     LIQUID_FFT_BENCHMARK_API(    66, LIQUID_FFT_FORWARD)
void benchmark_fft_68     LIQUID_FFT_BENCHMARK_API(    68, LIQUID_FFT_FORWARD)
void benchmark_fft_69     LIQUID_FFT_BENCHMARK_API(    69, LIQUID_FFT_FORWARD)
void benchmark_fft_70     LIQUID_FFT_BENCHMARK_API(    70, LIQUID_FFT_FORWARD)
void benchmark_fft_72     LIQUID_FFT_BENCHMARK_API(    72, LIQUID_FFT_FORWARD)
void benchmark_fft_74     LIQUID_FFT_BENCHMARK_API(    74, LIQUID_FFT_FORWARD)
void benchmark_fft_75     LIQUID_FFT_BENCHMARK_API(    75, LIQUID_FFT_FORWARD)
void benchmark_fft_76     LIQUID_FFT_BENCHMARK_API(    76, LIQUID_FFT_FORWARD)
void benchmark_fft_77     LIQUID_FFT_BENCHMARK_API(    77, LIQUID_FFT_FORWARD)
void benchmark_fft_78     LIQUID_FFT_BENCHMARK_API(    78, LIQUID_FFT_FORWARD)
void benchmark_fft_80     LIQUID_FFT_BENCHMARK_API(    80, LIQUID_FFT_FORWARD)
void benchmark_fft_81     LIQUID_FFT_BENCHMARK_API(    81, LIQUID_FFT_FORWARD)
void benchmark_fft_82     LIQUID_FFT_BENCHMARK_API(    82, LIQUID_FFT_FORWARD)
void benchmark_fft_84     LIQUID_FFT_BENCHMARK_API(    84, LIQUID_FFT_FORWARD)
void benchmark_fft_85     LIQUID_FFT_BENCHMARK_API(    85, LIQUID_FFT_FORWARD)
void benchmark_fft_86     LIQUID_FFT_BENCHMARK_API(    86, LIQUID_FFT_FORWARD)
void benchmark_fft_87     LIQUID_FFT_BENCHMARK_API(    87, LIQUID_FFT_FORWARD)
void benchmark_fft_88     LIQUID_FFT_BENCHMARK_API(    88, LIQUID_FFT_FORWARD)
void benchmark_fft_90     LIQUID_FFT_BENCHMARK_API(    90, LIQUID_FFT_FORWARD)
void benchmark_fft_91     LIQUID_FFT_BENCHMARK_API(    91, LIQUID_FFT_FORWARD)
void benchmark_fft_92     LIQUID_FFT_BENCHMARK_API(    92, LIQUID_FFT_FORWARD)
void benchmark_fft_93     LIQUID_FFT_BENCHMARK_API(    93, LIQUID_FFT_FORWARD)
void benchmark_fft_94     LIQUID_FFT_BENCHMARK_API(    94, LIQUID_FFT_FORWARD)
void benchmark_fft_95     LIQUID_FFT_BENCHMARK_API(    95, LIQUID_FFT_FORWARD)
void benchmark_fft_96     LIQUID_FFT_BENCHMARK_API(    96, LIQUID_FFT_FORWARD)
void benchmark_fft_98     LIQUID_FFT_BENCHMARK_API(    98, LIQUID_FFT_FORWARD)
void benchmark_fft_99     LIQUID_FFT_BENCHMARK_API(    99, LIQUID_FFT_FORWARD)
void benchmark_fft_100    LIQUID_FFT_BENCHMARK_API(   100, LIQUID_FFT_FORWARD)
void benchmark_fft_102    LIQUID_FFT_BENCHMARK_API(   102, LIQUID_FFT_FORWARD)
void benchmark_fft_104    LIQUID_FFT_BENCHMARK_API(   104, LIQUID_FFT_FORWARD)
void benchmark_fft_105    LIQUID_FFT_BENCHMARK_API(   105, LIQUID_FFT_FORWARD)
void benchmark_fft_106    LIQUID_FFT_BENCHMARK_API(   106, LIQUID_FFT_FORWARD)
void benchmark_fft_108    LIQUID_FFT_BENCHMARK_API(   108, LIQUID_FFT_FORWARD)
void benchmark_fft_110    LIQUID_FFT_BENCHMARK_API(   110, LIQUID_FFT_FORWARD)
void benchmark_fft_111    LIQUID_FFT_BENCHMARK_API(   111, LIQUID_FFT_FORWARD)
void benchmark_fft_112    LIQUID_FFT_BENCHMARK_API(   112, LIQUID_FFT_FORWARD)
void benchmark_fft_114    LIQUID_FFT_BENCHMARK_API(   114, LIQUID_FFT_FORWARD)
void benchmark_fft_115    LIQUID_FFT_BENCHMARK_API(   115, LIQUID_FFT_FORWARD)
void benchmark_fft_116    LIQUID_FFT_BENCHMARK_API(   116, LIQUID_FFT_FORWARD)
void benchmark_fft_117    LIQUID_FFT_BENCHMARK_API(   117, LIQUID_FFT_FORWARD)
void benchmark_fft_118    LIQUID_FFT_BENCHMARK_API(   118, LIQUID_FFT_FORWARD)
void benchmark_fft_119    LIQUID_FFT_BENCHMARK_API(   119, LIQUID_FFT_FORWARD)
void benchmark_fft_120    LIQUID_FFT_BENCHMARK_API(   120, LIQUID_FFT_FORWARD)
void benchmark_fft_121    LIQUID_FFT_BENCHMARK_API(   121, LIQUID_FFT_FORWARD)
void benchmark_fft_122    LIQUID_FFT_BENCHMARK_API(   122, LIQUID_FFT_FORWARD)
void benchmark_fft_123    LIQUID_FFT_BENCHMARK_API(   123, LIQUID_FFT_FORWARD)
void benchmark_fft_124    LIQUID_FFT_BENCHMARK_API(   124, LIQUID_FFT_FORWARD)
void benchmark_fft_125    LIQUID_FFT_BENCHMARK_API(   125, LIQUID_FFT_FORWARD)
void benchmark_fft_126    LIQUID_FFT_BENCHMARK_API(   126, LIQUID_FFT_FORWARD)
void benchmark_fft_129    LIQUID_FFT_BENCHMARK_API(   129, LIQUID_FFT_FORWARD)
void benchmark_fft_130    LIQUID_FFT_BENCHMARK_API(   130, LIQUID_FFT_FORWARD)
void benchmark_fft_132    LIQUID_FFT_BENCHMARK_API(   132, LIQUID_FFT_FORWARD)
void benchmark_fft_133    LIQUID_FFT_BENCHMARK_API(   133, LIQUID_FFT_FORWARD)
void benchmark_fft_134    LIQUID_FFT_BENCHMARK_API(   134, LIQUID_FFT_FORWARD)
void benchmark_fft_135    LIQUID_FFT_BENCHMARK_API(   135, LIQUID_FFT_FORWARD)
void benchmark_fft_136    LIQUID_FFT_BENCHMARK_API(   136, LIQUID_FFT_FORWARD)
void benchmark_fft_138    LIQUID_FFT_BENCHMARK_API(   138, LIQUID_FFT_FORWARD)
void benchmark_fft_140    LIQUID_FFT_BENCHMARK_API(   140, LIQUID_FFT_FORWARD)
void benchmark_fft_141    LIQUID_FFT_BENCHMARK_API(   141, LIQUID_FFT_FORWARD)
void benchmark_fft_142    LIQUID_FFT_BENCHMARK_API(   142, LIQUID_FFT_FORWARD)
void benchmark_fft_143    LIQUID_FFT_BENCHMARK_API(   143, LIQUID_FFT_FORWARD)
void benchmark_fft_144    LIQUID_FFT_BENCHMARK_API(   144, LIQUID_FFT_FORWARD)
void benchmark_fft_145    LIQUID_FFT_BENCHMARK_API(   145, LIQUID_FFT_FORWARD)
void benchmark_fft_146    LIQUID_FFT_BENCHMARK_API(   146, LIQUID_FFT_FORWARD)
void benchmark_fft_147    LIQUID_FFT_BENCHMARK_API(   147, LIQUID_FFT_FORWARD)
void benchmark_fft_148    LIQUID_FFT_BENCHMARK_API(   148, LIQUID_FFT_FORWARD)
void benchmark_fft_150    LIQUID_FFT_BENCHMARK_API(   150, LIQUID_FFT_FORWARD)
void benchmark_fft_152    LIQUID_FFT_BENCHMARK_API(   152, LIQUID_FFT_FORWARD)
void benchmark_fft_153    LIQUID_FFT_BENCHMARK_API(   153, LIQUID_FFT_FORWARD)
void benchmark_fft_154    LIQUID_FFT_BENCHMARK_API(   154, LIQUID_FFT_FORWARD)
void benchmark_fft_155    LIQUID_FFT_BENCHMARK_API(   155, LIQUID_FFT_FORWARD)
void benchmark_fft_156    LIQUID_FFT_BENCHMARK_API(   156, LIQUID_FFT_FORWARD)
void benchmark_fft_158    LIQUID_FFT_BENCHMARK_API(   158, LIQUID_FFT_FORWARD)
void benchmark_fft_159    LIQUID_FFT_BENCHMARK_API(   159, LIQUID_FFT_FORWARD)
void benchmark_fft_160    LIQUID_FFT_BENCHMARK_API(   160, LIQUID_FFT_FORWARD)
void benchmark_fft_161    LIQUID_FFT_BENCHMARK_API(   161, LIQUID_FFT_FORWARD)
void benchmark_fft_162    LIQUID_FFT_BENCHMARK_API(   162, LIQUID_FFT_FORWARD)
void benchmark_fft_164    LIQUID_FFT_BENCHMARK_API(   164, LIQUID_FFT_FORWARD)
void benchmark_fft_165    LIQUID_FFT_BENCHMARK_API(   165, LIQUID_FFT_FORWARD)
void benchmark_fft_166    LIQUID_FFT_BENCHMARK_API(   166, LIQUID_FFT_FORWARD)
void benchmark_fft_168    LIQUID_FFT_BENCHMARK_API(   168, LIQUID_FFT_FORWARD)
void benchmark_fft_169    LIQUID_FFT_BENCHMARK_API(   169, LIQUID_FFT_FORWARD)
void benchmark_fft_170    LIQUID_FFT_BENCHMARK_API(   170, LIQUID_FFT_FORWARD)
void benchmark_fft_171    LIQUID_FFT_BENCHMARK_API(   171, LIQUID_FFT_FORWARD)
void benchmark_fft_172    LIQUID_FFT_BENCHMARK_API(   172, LIQUID_FFT_FORWARD)
void benchmark_fft_174    LIQUID_FFT_BENCHMARK_API(   174, LIQUID_FFT_FORWARD)
void benchmark_fft_175    LIQUID_FFT_BENCHMARK_API(   175, LIQUID_FFT_FORWARD)
void benchmark_fft_176    LIQUID_FFT_BENCHMARK_API(   176, LIQUID_FFT_FORWARD)
void benchmark_fft_177    LIQUID_FFT_BENCHMARK_API(   177, LIQUID_FFT_FORWARD)
void benchmark_fft_178    LIQUID_FFT_BENCHMARK_API(   178, LIQUID_FFT_FORWARD)
void benchmark_fft_180    LIQUID_FFT_BENCHMARK_API(   180, LIQUID_FFT_FORWARD)
void benchmark_fft_182    LIQUID_FFT_BENCHMARK_API(   182, LIQUID_FFT_FORWARD)
void benchmark_fft_183    LIQUID_FFT_BENCHMARK_API(   183, LIQUID_FFT_FORWARD)
void benchmark_fft_184    LIQUID_FFT_BENCHMARK_API(   184, LIQUID_FFT_FORWARD)
void benchmark_fft_185    LIQUID_FFT_BENCHMARK_API(   185, LIQUID_FFT_FORWARD)
void benchmark_fft_186    LIQUID_FFT_BENCHMARK_API(   186, LIQUID_FFT_FORWARD)
void benchmark_fft_187    LIQUID_FFT_BENCHMARK_API(   187, LIQUID_FFT_FORWARD)
void benchmark_fft_188    LIQUID_FFT_BENCHMARK_API(   188, LIQUID_FFT_FORWARD)
void benchmark_fft_189    LIQUID_FFT_BENCHMARK_API(   189, LIQUID_FFT_FORWARD)
void benchmark_fft_190    LIQUID_FFT_BENCHMARK_API(   190, LIQUID_FFT_FORWARD)
void benchmark_fft_192    LIQUID_FFT_BENCHMARK_API(   192, LIQUID_FFT_FORWARD)
void benchmark_fft_194    LIQUID_FFT_BENCHMARK_API(   194, LIQUID_FFT_FORWARD)
void benchmark_fft_195    LIQUID_FFT_BENCHMARK_API(   195, LIQUID_FFT_FORWARD)
void benchmark_fft_196    LIQUID_FFT_BENCHMARK_API(   196, LIQUID_FFT_FORWARD)
void benchmark_fft_198    LIQUID_FFT_BENCHMARK_API(   198, LIQUID_FFT_FORWARD)
void benchmark_fft_200    LIQUID_FFT_BENCHMARK_API(   200, LIQUID_FFT_FORWARD)
void benchmark_fft_201    LIQUID_FFT_BENCHMARK_API(   201, LIQUID_FFT_FORWARD)
void benchmark_fft_202    LIQUID_FFT_BENCHMARK_API(   202, LIQUID_FFT_FORWARD)
void benchmark_fft_203    LIQUID_FFT_BENCHMARK_API(   203, LIQUID_FFT_FORWARD)
void benchmark_fft_204    LIQUID_FFT_BENCHMARK_API(   204, LIQUID_FFT_FORWARD)
void benchmark_fft_205    LIQUID_FFT_BENCHMARK_API(   205, LIQUID_FFT_FORWARD)
void benchmark_fft_206    LIQUID_FFT_BENCHMARK_API(   206, LIQUID_FFT_FORWARD)
void benchmark_fft_207    LIQUID_FFT_BENCHMARK_API(   207, LIQUID_FFT_FORWARD)
void benchmark_fft_208    LIQUID_FFT_BENCHMARK_API(   208, LIQUID_FFT_FORWARD)
void benchmark_fft_209    LIQUID_FFT_BENCHMARK_API(   209, LIQUID_FFT_FORWARD)
void benchmark_fft_210    LIQUID_FFT_BENCHMARK_API(   210, LIQUID_FFT_FORWARD)
void benchmark_fft_212    LIQUID_FFT_BENCHMARK_API(   212, LIQUID_FFT_FORWARD)
void benchmark_fft_213    LIQUID_FFT_BENCHMARK_API(   213, LIQUID_FFT_FORWARD)
void benchmark_fft_214    LIQUID_FFT_BENCHMARK_API(   214, LIQUID_FFT_FORWARD)
void benchmark_fft_215    LIQUID_FFT_BENCHMARK_API(   215, LIQUID_FFT_FORWARD)
void benchmark_fft_216    LIQUID_FFT_BENCHMARK_API(   216, LIQUID_FFT_FORWARD)
void benchmark_fft_217    LIQUID_FFT_BENCHMARK_API(   217, LIQUID_FFT_FORWARD)
void benchmark_fft_218    LIQUID_FFT_BENCHMARK_API(   218, LIQUID_FFT_FORWARD)
void benchmark_fft_219    LIQUID_FFT_BENCHMARK_API(   219, LIQUID_FFT_FORWARD)
void benchmark_fft_220    LIQUID_FFT_BENCHMARK_API(   220, LIQUID_FFT_FORWARD)
void benchmark_fft_221    LIQUID_FFT_BENCHMARK_API(   221, LIQUID_FFT_FORWARD)
void benchmark_fft_222    LIQUID_FFT_BENCHMARK_API(   222, LIQUID_FFT_FORWARD)
void benchmark_fft_224    LIQUID_FFT_BENCHMARK_API(   224, LIQUID_FFT_FORWARD)
void benchmark_fft_225    LIQUID_FFT_BENCHMARK_API(   225, LIQUID_FFT_FORWARD)
void benchmark_fft_226    LIQUID_FFT_BENCHMARK_API(   226, LIQUID_FFT_FORWARD)
void benchmark_fft_228    LIQUID_FFT_BENCHMARK_API(   228, LIQUID_FFT_FORWARD)
void benchmark_fft_230    LIQUID_FFT_BENCHMARK_API(   230, LIQUID_FFT_FORWARD)
void benchmark_fft_231    LIQUID_FFT_BENCHMARK_API(   231, LIQUID_FFT_FORWARD)
void benchmark_fft_232    LIQUID_FFT_BENCHMARK_API(   232, LIQUID_FFT_FORWARD)
void benchmark_fft_234    LIQUID_FFT_BENCHMARK_API(   234, LIQUID_FFT_FORWARD)
void benchmark_fft_235    LIQUID_FFT_BENCHMARK_API(   235, LIQUID_FFT_FORWARD)
void benchmark_fft_236    LIQUID_FFT_BENCHMARK_API(   236, LIQUID_FFT_FORWARD)
void benchmark_fft_237    LIQUID_FFT_BENCHMARK_API(   237, LIQUID_FFT_FORWARD)
void benchmark_fft_238    LIQUID_FFT_BENCHMARK_API(   238, LIQUID_FFT_FORWARD)
void benchmark_fft_240    LIQUID_FFT_BENCHMARK_API(   240, LIQUID_FFT_FORWARD)
void benchmark_fft_242    LIQUID_FFT_BENCHMARK_API(   242, LIQUID_FFT_FORWARD)
void benchmark_fft_243    LIQUID_FFT_BENCHMARK_API(   243, LIQUID_FFT_FORWARD)
void benchmark_fft_244    LIQUID_FFT_BENCHMARK_API(   244, LIQUID_FFT_FORWARD)
void benchmark_fft_245    LIQUID_FFT_BENCHMARK_API(   245, LIQUID_FFT_FORWARD)
void benchmark_fft_246    LIQUID_FFT_BENCHMARK_API(   246, LIQUID_FFT_FORWARD)
void benchmark_fft_247    LIQUID_FFT_BENCHMARK_API(   247, LIQUID_FFT_FORWARD)
void benchmark_fft_248    LIQUID_FFT_BENCHMARK_API(   248, LIQUID_FFT_FORWARD)
void benchmark_fft_249    LIQUID_FFT_BENCHMARK_API(   249, LIQUID_FFT_FORWARD)
void benchmark_fft_250    LIQUID_FFT_BENCHMARK_API(   250, LIQUID_FFT_FORWARD)
void benchmark_fft_252    LIQUID_FFT_BENCHMARK_API(   252, LIQUID_FFT_FORWARD)
void benchmark_fft_253    LIQUID_FFT_BENCHMARK_API(   253, LIQUID_FFT_FORWARD)
void benchmark_fft_254    LIQUID_FFT_BENCHMARK_API(   254, LIQUID_FFT_FORWARD)
void benchmark_fft_255    LIQUID_FFT_BENCHMARK_API(   255, LIQUID_FFT_FORWARD)
