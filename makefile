# Copyright (c) 2007, 2008, 2009, 2010, 2011, 2012, 2013 Joseph Gaeddert
#
# This file is part of liquid.
#
# liquid is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# liquid is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with liquid.  If not, see <http://www.gnu.org/licenses/>.

# 
# Makefile for liquid-dsp libraries
#
# Targets:
#    all                 :   dynamic shared-library object (e.g. libliquid.so)
#    install             :   install the dynamic shared library object and
#                            header file(s)
#    uninstall           :   uninstall the library and header file(s)
#    clean               :   clean all targets (bench, check, examples, etc)
#    distclean           :   removes everything except the originally distributed files
#    check               :   build and run autotest program
#    bench               :   build and run benchmarking program
#    examples            :   build all examples
#    sandbox             :   build all sandbox examples
#    world               :   build absolutely everything (but don't install)
#
#    clean-modules       :   clean all modules
#    clean-examples      :   clean examples programs
#    clean-sandbox       :   clean sandbox programs
#    clean-check         :   clean autotest program
#    clean-bench         :   clean benchmark program
#    clean-doc           :   clean documentation
#

# autoconf initialization macros
NAME      := liquid-dsp
VERSION   := 1.2.0
BUGREPORT := joseph@liquidsdr.org

# paths
srcdir = .
prefix = /home/mihai/anul_IV/sem1/app/liquid-dsp/installation
exec_prefix = ${prefix}

include_dirs	:= . include
vpath %.h $(include_dirs)
modulesdir = src

# programs
CC = gcc
MV	:= mv -f
RM	:= rm -f
SED	:= /bin/sed
GREP	:= /bin/grep
AR	:= ar
RANLIB	:= ranlib

# flags
INCLUDE_CFLAGS	= $(addprefix -I ,$(include_dirs))
CONFIG_CFLAGS	=     -mmmx -msse -msse2
# -g : debugging info
CFLAGS		+= $(INCLUDE_CFLAGS) -g -O2 -Wall -fPIC $(CONFIG_CFLAGS)
LDFLAGS		+= -lm -lc 
ARFLAGS		= r
PATHSEP		= /

# preprocessor user defined variables (with -D)
# DEFINES = -DDEBUG -DRANDOM_DATA

# 
# liquid headers
#
headers_install	:= liquid.h 
headers		:= $(headers_install) liquid.internal.h
include_headers	:= $(addprefix include/,$(headers))


## 
## liquid-dsp modules
##

all:

# additional targets to clean
extra_clean :=

# additional autotest objects
autotest_extra_obj :=

# additional benchmark objects
benchmark_extra_obj :=

#
# MODULE : agc - automatic gain control
#

# object files
agc_objects =							\
	src/agc/src/agc_crcf.o					\
	src/agc/src/agc_rrrf.o					\

# explicit targets and dependencies

src/agc/src/agc_crcf.o : %.o : %.c src/agc/src/agc.c $(headers)

src/agc/src/agc_rrrf.o : %.o : %.c src/agc/src/agc.c $(headers)

# autotests
agc_autotests :=						\
	src/agc/tests/agc_crcf_autotest.c			\

# benchmarks
agc_benchmarks :=						\
	src/agc/bench/agc_crcf_benchmark.c			\

#
# MODULE : audio
#

# described below
audio_objects :=						\
	src/audio/src/cvsd.o					\

src/cvsd/src/cvsd.o : %.o : %.c $(headers)


audio_autotests :=						\
	src/audio/tests/cvsd_autotest.c				\

audio_benchmarks :=						\
	src/audio/bench/cvsd_benchmark.c			\


# 
# MODULE : buffer
# 

buffer_objects :=						\
	src/buffer/src/bufferf.o				\
	src/buffer/src/buffercf.o				\

#	src/buffer/src/bufferui.c				\

buffer_includes :=						\
	src/buffer/src/buffer.c					\
	src/buffer/src/wdelay.c					\
	src/buffer/src/window.c					\

src/buffer/src/bufferf.o : %.o : %.c $(headers) $(buffer_includes)

src/buffer/src/buffercf.o : %.o : %.c $(headers) $(buffer_includes)

src/buffer/src/bufferui.o : %.o : %.c $(headers) $(buffer_includes)


buffer_autotests :=						\
	src/buffer/tests/cbuffer_autotest.c			\
	src/buffer/tests/sbuffer_autotest.c			\
	src/buffer/tests/wdelay_autotest.c			\
	src/buffer/tests/window_autotest.c			\

buffer_benchmarks :=						\
	src/buffer/bench/window_push_benchmark.c		\
	src/buffer/bench/window_read_benchmark.c		\

# 
# MODULE : dotprod
#
dotprod_objects :=						\
	src/dotprod/src/dotprod_cccf.mmx.o                        src/dotprod/src/dotprod_crcf.mmx.o                        src/dotprod/src/dotprod_rrrf.mmx.o                        src/dotprod/src/sumsq.mmx.o						\

src/dotprod/src/dotprod_cccf.o : %.o : %.c $(headers) src/dotprod/src/dotprod.c

src/dotprod/src/dotprod_crcf.o : %.o : %.c $(headers) src/dotprod/src/dotprod.c

src/dotprod/src/dotprod_rrrf.o : %.o : %.c $(headers) src/dotprod/src/dotprod.c

src/dotprod/src/sumsq.o : %.o : %.c $(headers)

# specific machine architectures

# AltiVec
src/dotprod/src/dotprod_rrrf.av.o : %.o : %.c $(headers)

# MMX/SSE2
src/dotprod/src/dotprod_rrrf.mmx.o : %.o : %.c $(headers)
src/dotprod/src/dotprod_crcf.mmx.o : %.o : %.c $(headers)
src/dotprod/src/dotprod_cccf.mmx.o : %.o : %.c $(headers)

src/dotprod/src/sumsq.mmx.o : %.o : %.c $(headers)

# SSE4.1/2
src/dotprod/src/dotprod_rrrf.sse4.o : %.o : %.c $(headers)

dotprod_autotests :=						\
	src/dotprod/tests/dotprod_rrrf_autotest.c		\
	src/dotprod/tests/dotprod_crcf_autotest.c		\
	src/dotprod/tests/dotprod_cccf_autotest.c		\
	src/dotprod/tests/sumsqf_autotest.c			\
	src/dotprod/tests/sumsqcf_autotest.c			\

dotprod_benchmarks :=						\
	src/dotprod/bench/dotprod_cccf_benchmark.c		\
	src/dotprod/bench/dotprod_crcf_benchmark.c		\
	src/dotprod/bench/dotprod_rrrf_benchmark.c		\
	src/dotprod/bench/sumsqf_benchmark.c			\
	src/dotprod/bench/sumsqcf_benchmark.c			\

# 
# MODULE : equalization
#
equalization_objects :=						\
	src/equalization/src/equalizer_cccf.o			\
	src/equalization/src/equalizer_rrrf.o			\


$(equalization_objects) : %.o : %.c $(headers) src/equalization/src/eqlms.c src/equalization/src/eqrls.c


# autotests
equalization_autotests :=					\
	src/equalization/tests/eqrls_rrrf_autotest.c		\


# benchmarks
equalization_benchmarks :=					\
	src/equalization/bench/eqlms_cccf_benchmark.c		\
	src/equalization/bench/eqrls_cccf_benchmark.c		\

# 
# MODULE : fec - forward error-correction
#
fec_objects :=							\
	src/fec/src/crc.o					\
	src/fec/src/fec.o					\
	src/fec/src/fec_conv.o					\
	src/fec/src/fec_conv_poly.o				\
	src/fec/src/fec_conv_pmatrix.o				\
	src/fec/src/fec_conv_punctured.o			\
	src/fec/src/fec_golay2412.o				\
	src/fec/src/fec_hamming74.o				\
	src/fec/src/fec_hamming84.o				\
	src/fec/src/fec_hamming128.o				\
	src/fec/src/fec_hamming1511.o				\
	src/fec/src/fec_hamming3126.o				\
	src/fec/src/fec_hamming128_gentab.o			\
	src/fec/src/fec_pass.o					\
	src/fec/src/fec_rep3.o					\
	src/fec/src/fec_rep5.o					\
	src/fec/src/fec_rs.o					\
	src/fec/src/fec_secded2216.o				\
	src/fec/src/fec_secded3932.o				\
	src/fec/src/fec_secded7264.o				\
	src/fec/src/interleaver.o				\
	src/fec/src/packetizer.o				\
	src/fec/src/sumproduct.o				\


# list explicit targets and dependencies here
$(fec_objects) : %.o : %.c $(headers)

# autotests
fec_autotests :=						\
	src/fec/tests/crc_autotest.c				\
	src/fec/tests/fec_autotest.c				\
	src/fec/tests/fec_soft_autotest.c			\
	src/fec/tests/fec_golay2412_autotest.c			\
	src/fec/tests/fec_hamming74_autotest.c			\
	src/fec/tests/fec_hamming84_autotest.c			\
	src/fec/tests/fec_hamming128_autotest.c			\
	src/fec/tests/fec_hamming1511_autotest.c		\
	src/fec/tests/fec_hamming3126_autotest.c		\
	src/fec/tests/fec_reedsolomon_autotest.c		\
	src/fec/tests/fec_rep3_autotest.c			\
	src/fec/tests/fec_rep5_autotest.c			\
	src/fec/tests/fec_secded2216_autotest.c			\
	src/fec/tests/fec_secded3932_autotest.c			\
	src/fec/tests/fec_secded7264_autotest.c			\
	src/fec/tests/interleaver_autotest.c			\
	src/fec/tests/packetizer_autotest.c			\


# benchmarks
fec_benchmarks :=						\
	src/fec/bench/crc_benchmark.c				\
	src/fec/bench/fec_encode_benchmark.c			\
	src/fec/bench/fec_decode_benchmark.c			\
	src/fec/bench/fecsoft_decode_benchmark.c		\
	src/fec/bench/sumproduct_benchmark.c			\
	src/fec/bench/interleaver_benchmark.c			\
	src/fec/bench/packetizer_decode_benchmark.c		\

# 
# MODULE : fft - fast Fourier transforms, discrete sine/cosine transforms, etc.
#

fft_objects :=							\
	src/fft/src/fftf.o					\
	src/fft/src/asgram.o					\
	src/fft/src/spgram.o					\
	src/fft/src/fft_utilities.o				\

# explicit targets and dependencies
fft_includes :=							\
	src/fft/src/fft_common.c				\
	src/fft/src/fft_dft.c					\
	src/fft/src/fft_radix2.c				\
	src/fft/src/fft_mixed_radix.c				\
	src/fft/src/fft_rader.c					\
	src/fft/src/fft_rader2.c				\
	src/fft/src/fft_r2r_1d.c				\

src/fft/src/fftf.o : %.o : %.c $(headers) $(fft_includes)

src/fft/src/asgram.o : %.o : %.c $(headers)

src/fft/src/dct.o : %.o : %.c $(headers)

src/fft/src/fftf.o : %.o : %.c $(headers)

src/fft/src/fft_utilities.o : %.o : %.c $(headers)

src/fft/src/mdct.o : %.o : %.c $(headers)

# fft autotest scripts
fft_autotests :=						\
	src/fft/tests/fft_small_autotest.c			\
	src/fft/tests/fft_radix2_autotest.c			\
	src/fft/tests/fft_composite_autotest.c			\
	src/fft/tests/fft_prime_autotest.c			\
	src/fft/tests/fft_r2r_autotest.c			\
	src/fft/tests/fft_shift_autotest.c			\

# additional autotest objects
autotest_extra_obj +=						\
	src/fft/tests/fft_runtest.o				\
	src/fft/tests/data/fft_data_2.o				\
	src/fft/tests/data/fft_data_3.o				\
	src/fft/tests/data/fft_data_4.o				\
	src/fft/tests/data/fft_data_5.o				\
	src/fft/tests/data/fft_data_6.o				\
	src/fft/tests/data/fft_data_7.o				\
	src/fft/tests/data/fft_data_8.o				\
	src/fft/tests/data/fft_data_9.o				\
	src/fft/tests/data/fft_data_10.o			\
	src/fft/tests/data/fft_data_16.o			\
	src/fft/tests/data/fft_data_17.o			\
	src/fft/tests/data/fft_data_20.o			\
	src/fft/tests/data/fft_data_21.o			\
	src/fft/tests/data/fft_data_22.o			\
	src/fft/tests/data/fft_data_24.o			\
	src/fft/tests/data/fft_data_26.o			\
	src/fft/tests/data/fft_data_30.o			\
	src/fft/tests/data/fft_data_32.o			\
	src/fft/tests/data/fft_data_35.o			\
	src/fft/tests/data/fft_data_36.o			\
	src/fft/tests/data/fft_data_43.o			\
	src/fft/tests/data/fft_data_48.o			\
	src/fft/tests/data/fft_data_63.o			\
	src/fft/tests/data/fft_data_64.o			\
	src/fft/tests/data/fft_data_79.o			\
	src/fft/tests/data/fft_data_92.o			\
	src/fft/tests/data/fft_data_96.o			\
	src/fft/tests/data/fft_data_120.o			\
	src/fft/tests/data/fft_data_130.o			\
	src/fft/tests/data/fft_data_157.o			\
	src/fft/tests/data/fft_data_192.o			\
	src/fft/tests/data/fft_data_317.o			\
	src/fft/tests/data/fft_data_509.o			\
	src/fft/tests/data/fft_r2rdata_8.o			\
	src/fft/tests/data/fft_r2rdata_27.o			\
	src/fft/tests/data/fft_r2rdata_32.o			\

# fft benchmark scripts
fft_benchmarks :=						\
	src/fft/bench/fft_composite_benchmark.c			\
	src/fft/bench/fft_prime_benchmark.c			\
	src/fft/bench/fft_radix2_benchmark.c			\
	src/fft/bench/fft_r2r_benchmark.c			\

# additional benchmark objects
benchmark_extra_obj :=						\
	src/fft/bench/fft_runbench.o				\

#
# MODULE : filter
#

filter_objects :=						\
	src/filter/src/bessel.o					\
	src/filter/src/butter.o					\
	src/filter/src/cheby1.o					\
	src/filter/src/cheby2.o					\
	src/filter/src/ellip.o					\
	src/filter/src/filter_rrrf.o				\
	src/filter/src/filter_crcf.o				\
	src/filter/src/filter_cccf.o				\
	src/filter/src/firdes.o					\
	src/filter/src/firdespm.o				\
	src/filter/src/fnyquist.o				\
	src/filter/src/gmsk.o					\
	src/filter/src/group_delay.o				\
	src/filter/src/hM3.o					\
	src/filter/src/iirdes.pll.o				\
	src/filter/src/iirdes.o					\
	src/filter/src/lpc.o					\
	src/filter/src/rcos.o					\
	src/filter/src/rkaiser.o				\
	src/filter/src/rrcos.o					\


# list explicit targets and dependencies here
filter_includes :=						\
	src/filter/src/firdecim.c				\
	src/filter/src/firfarrow.c				\
	src/filter/src/firfilt.c				\
	src/filter/src/firhilb.c				\
	src/filter/src/firinterp.c				\
	src/filter/src/firpfb.c					\
	src/filter/src/iirdecim.c				\
	src/filter/src/iirfilt.c				\
	src/filter/src/iirfiltsos.c				\
	src/filter/src/iirinterp.c				\
	src/filter/src/msresamp.c				\
	src/filter/src/resamp.c					\
	src/filter/src/resamp2.c				\
	src/filter/src/symsync.c				\

src/filter/src/bessel.o : %.o : %.c $(headers)

src/filter/src/bessel.o : %.o : %.c $(headers)

src/filter/src/butter.o : %.o : %.c $(headers)

src/filter/src/cheby1.o : %.o : %.c $(headers)

src/filter/src/cheby2.o : %.o : %.c $(headers)

src/filter/src/ellip.o : %.o : %.c $(headers)

src/filter/src/filter_rrrf.o : %.o : %.c $(headers) $(filter_includes)

src/filter/src/filter_crcf.o : %.o : %.c $(headers) $(filter_includes)

src/filter/src/filter_cccf.o : %.o : %.c $(headers) $(filter_includes)

src/filter/src/firdes.o : %.o : %.c $(headers)

src/filter/src/firdespm.o : %.o : %.c $(headers)

src/filter/src/group_delay.o : %.o : %.c $(headers)

src/filter/src/hM3.o : %.o : %.c $(headers)

src/filter/src/iirdes.pll.o : %.o : %.c $(headers)

src/filter/src/iirdes.o : %.o : %.c $(headers)

src/filter/src/lpc.o : %.o : %.c $(headers)

src/filter/src/rcos.o : %.o : %.c $(headers)

src/filter/src/rkaiser.o : %.o : %.c $(headers)

src/filter/src/rrcos.o : %.o : %.c $(headers)


filter_autotests :=						\
	src/filter/tests/filter_crosscorr_autotest.c		\
	src/filter/tests/firdecim_xxxf_autotest.c		\
	src/filter/tests/firdes_autotest.c			\
	src/filter/tests/firdespm_autotest.c			\
	src/filter/tests/firfilt_xxxf_autotest.c		\
	src/filter/tests/firhilb_autotest.c			\
	src/filter/tests/firinterp_autotest.c			\
	src/filter/tests/firpfb_autotest.c			\
	src/filter/tests/groupdelay_autotest.c			\
	src/filter/tests/iirdes_autotest.c			\
	src/filter/tests/iirfilt_xxxf_autotest.c		\
	src/filter/tests/iirfiltsos_rrrf_autotest.c		\
	src/filter/tests/msresamp_crcf_autotest.c		\
	src/filter/tests/resamp_crcf_autotest.c			\
	src/filter/tests/resamp2_crcf_autotest.c		\

# additional autotest objects
autotest_extra_obj +=						\
	src/filter/tests/firdecim_runtest.o			\
								\
	src/filter/tests/data/firdecim_rrrf_data_M2h4x20.o	\
	src/filter/tests/data/firdecim_crcf_data_M2h4x20.o	\
	src/filter/tests/data/firdecim_cccf_data_M2h4x20.o	\
								\
	src/filter/tests/data/firdecim_rrrf_data_M3h7x30.o	\
	src/filter/tests/data/firdecim_crcf_data_M3h7x30.o	\
	src/filter/tests/data/firdecim_cccf_data_M3h7x30.o	\
								\
	src/filter/tests/data/firdecim_rrrf_data_M4h13x40.o	\
	src/filter/tests/data/firdecim_crcf_data_M4h13x40.o	\
	src/filter/tests/data/firdecim_cccf_data_M4h13x40.o	\
								\
	src/filter/tests/data/firdecim_rrrf_data_M5h23x50.o	\
	src/filter/tests/data/firdecim_crcf_data_M5h23x50.o	\
	src/filter/tests/data/firdecim_cccf_data_M5h23x50.o	\
								\
	src/filter/tests/firfilt_runtest.o			\
								\
	src/filter/tests/data/firfilt_rrrf_data_h4x8.o		\
	src/filter/tests/data/firfilt_crcf_data_h4x8.o		\
	src/filter/tests/data/firfilt_cccf_data_h4x8.o		\
								\
	src/filter/tests/data/firfilt_rrrf_data_h7x16.o		\
	src/filter/tests/data/firfilt_crcf_data_h7x16.o		\
	src/filter/tests/data/firfilt_cccf_data_h7x16.o		\
								\
	src/filter/tests/data/firfilt_rrrf_data_h13x32.o	\
	src/filter/tests/data/firfilt_crcf_data_h13x32.o	\
	src/filter/tests/data/firfilt_cccf_data_h13x32.o	\
								\
	src/filter/tests/data/firfilt_rrrf_data_h23x64.o	\
	src/filter/tests/data/firfilt_crcf_data_h23x64.o	\
	src/filter/tests/data/firfilt_cccf_data_h23x64.o	\
								\
	src/filter/tests/iirfilt_runtest.o			\
								\
	src/filter/tests/data/iirfilt_rrrf_data_h3x64.o		\
	src/filter/tests/data/iirfilt_crcf_data_h3x64.o		\
	src/filter/tests/data/iirfilt_cccf_data_h3x64.o		\
								\
	src/filter/tests/data/iirfilt_rrrf_data_h5x64.o		\
	src/filter/tests/data/iirfilt_crcf_data_h5x64.o		\
	src/filter/tests/data/iirfilt_cccf_data_h5x64.o		\
								\
	src/filter/tests/data/iirfilt_rrrf_data_h7x64.o		\
	src/filter/tests/data/iirfilt_crcf_data_h7x64.o		\
	src/filter/tests/data/iirfilt_cccf_data_h7x64.o		\

filter_benchmarks :=						\
	src/filter/bench/firdecim_benchmark.c			\
	src/filter/bench/firhilb_benchmark.c			\
	src/filter/bench/firinterp_crcf_benchmark.c		\
	src/filter/bench/firfilt_crcf_benchmark.c		\
	src/filter/bench/iirfilt_crcf_benchmark.c		\
	src/filter/bench/resamp_crcf_benchmark.c		\
	src/filter/bench/resamp2_crcf_benchmark.c		\
	src/filter/bench/symsync_crcf_benchmark.c		\

# 
# MODULE : framing
#

framing_objects :=						\
	src/framing/src/bpacketgen.o				\
	src/framing/src/bpacketsync.o				\
	src/framing/src/bpresync_cccf.o				\
	src/framing/src/bsync_rrrf.o				\
	src/framing/src/bsync_crcf.o				\
	src/framing/src/bsync_cccf.o				\
	src/framing/src/detector_cccf.o				\
	src/framing/src/framesyncstats.o			\
	src/framing/src/framegen64.o				\
	src/framing/src/framesync64.o				\
	src/framing/src/flexframegen.o				\
	src/framing/src/flexframesync.o				\
	src/framing/src/gmskframegen.o				\
	src/framing/src/gmskframesync.o				\
	src/framing/src/ofdmflexframegen.o			\
	src/framing/src/ofdmflexframesync.o			\
	src/framing/src/presync_cccf.o				\


# list explicit targets and dependencies here

src/framing/src/bpacketgen.o : %.o : %.c $(headers)

src/framing/src/bpacketsync.o : %.o : %.c $(headers)

src/framing/src/bpresync_cccf.o : %.o : %.c $(headers) src/framing/src/bpresync.c

src/framing/src/bsync_rrrf.o : %.o : %.c $(headers) src/framing/src/bsync.c

src/framing/src/bsync_crcf.o : %.o : %.c $(headers) src/framing/src/bsync.c

src/framing/src/bsync_cccf.o : %.o : %.c $(headers) src/framing/src/bsync.c

src/framing/src/detector_cccf.o : %.o : %.c $(headers)

src/framing/src/framesyncstats.o : %.o : %.c $(headers)

src/framing/src/framegen64.o : %.o : %.c $(headers)

src/framing/src/framesync64.o : %.o : %.c $(headers)

src/framing/src/flexframegen.o : %.o : %.c $(headers)

src/framing/src/flexframesync.o : %.o : %.c $(headers)

src/framing/src/ofdmflexframegen.o : %.o : %.c $(headers)

src/framing/src/ofdmflexframesync.o : %.o : %.c $(headers)

src/framing/src/presync_cccf.o : %.o : %.c $(headers) src/framing/src/presync.c


framing_autotests :=						\
	src/framing/tests/bpacketsync_autotest.c		\
	src/framing/tests/bsync_autotest.c			\
	src/framing/tests/detector_autotest.c			\
	src/framing/tests/framesync64_autotest.c		\


framing_benchmarks :=						\
	src/framing/bench/presync_benchmark.c			\
	src/framing/bench/bpacketsync_benchmark.c		\
	src/framing/bench/bpresync_benchmark.c			\
	src/framing/bench/bsync_benchmark.c			\
	src/framing/bench/detector_benchmark.c			\
	src/framing/bench/flexframesync_benchmark.c		\
	src/framing/bench/framesync64_benchmark.c		\
	src/framing/bench/gmskframesync_benchmark.c		\


# 
# MODULE : math
#

math_objects :=							\
	src/math/src/poly.o					\
	src/math/src/polyc.o					\
	src/math/src/polyf.o					\
	src/math/src/polycf.o					\
	src/math/src/math.o					\
	src/math/src/math.bessel.o				\
	src/math/src/math.gamma.o				\
	src/math/src/math.complex.o				\
	src/math/src/math.trig.o				\
	src/math/src/modular_arithmetic.o			\


poly_includes :=						\
	src/math/src/poly.common.c				\
	src/math/src/poly.expand.c				\
	src/math/src/poly.findroots.c				\
	src/math/src/poly.lagrange.c				\

src/math/src/poly.o : %.o : %.c $(headers) $(poly_includes)

src/math/src/polyc.o : %.o : %.c $(headers) $(poly_includes)

src/math/src/polyf.o : %.o : %.c $(headers) $(poly_includes)

src/math/src/polycf.o : %.o : %.c $(headers) $(poly_includes)

src/math/src/math.o : %.o : %.c $(headers)

src/math/src/math.bessel.o : %.o : %.c $(headers)

src/math/src/math.gamma.o : %.o : %.c $(headers)

src/math/src/math.complex.o : %.o : %.c $(headers)

src/math/src/math.trig.o : %.o : %.c $(headers)

src/math/src/modular_arithmetic.o : %.o : %.c $(headers)


math_autotests :=						\
	src/math/tests/kbd_autotest.c				\
	src/math/tests/math_autotest.c				\
	src/math/tests/math_bessel_autotest.c			\
	src/math/tests/math_gamma_autotest.c			\
	src/math/tests/math_complex_autotest.c			\
	src/math/tests/polynomial_autotest.c			\


math_benchmarks :=						\
	src/math/bench/polyfit_benchmark.c			\


#
# MODULE : matrix
#

matrix_objects :=						\
	src/matrix/src/matrix.o					\
	src/matrix/src/matrixf.o				\
	src/matrix/src/matrixc.o				\
	src/matrix/src/matrixcf.o				\
	src/matrix/src/smatrix.common.o				\
	src/matrix/src/smatrixb.o				\
	src/matrix/src/smatrixf.o				\
	src/matrix/src/smatrixi.o				\


matrix_includes :=						\
	src/matrix/src/matrix.base.c				\
	src/matrix/src/matrix.cgsolve.c				\
	src/matrix/src/matrix.chol.c				\
	src/matrix/src/matrix.gramschmidt.c			\
	src/matrix/src/matrix.inv.c				\
	src/matrix/src/matrix.linsolve.c			\
	src/matrix/src/matrix.ludecomp.c			\
	src/matrix/src/matrix.qrdecomp.c			\
	src/matrix/src/matrix.math.c				\

src/matrix/src/matrix.o : %.o : %.c $(headers) $(matrix_includes)

src/matrix/src/matrixc.o : %.o : %.c $(headers) $(matrix_includes)

src/matrix/src/matrixf.o : %.o : %.c $(headers) $(matrix_includes)

src/matrix/src/matrixcf.o : %.o : %.c $(headers) $(matrix_includes)

src/matrix/src/smatrixb.o: %.o : %.c $(headers) src/matrix/src/smatrix.c

src/matrix/src/smatrixf.o: %.o : %.c $(headers) src/matrix/src/smatrix.c

src/matrix/src/smatrixi.o: %.o : %.c $(headers) src/matrix/src/smatrix.c


# matrix autotest scripts
matrix_autotests :=						\
	src/matrix/tests/matrixcf_autotest.c			\
	src/matrix/tests/matrixf_autotest.c			\
	src/matrix/tests/smatrixb_autotest.c			\
	src/matrix/tests/smatrixf_autotest.c			\
	src/matrix/tests/smatrixi_autotest.c			\

# additional autotest objects
autotest_extra_obj +=						\
	src/matrix/tests/data/matrixf_data_add.o		\
	src/matrix/tests/data/matrixf_data_aug.o		\
	src/matrix/tests/data/matrixf_data_cgsolve.o		\
	src/matrix/tests/data/matrixf_data_chol.o		\
	src/matrix/tests/data/matrixf_data_gramschmidt.o	\
	src/matrix/tests/data/matrixf_data_inv.o		\
	src/matrix/tests/data/matrixf_data_linsolve.o		\
	src/matrix/tests/data/matrixf_data_ludecomp.o		\
	src/matrix/tests/data/matrixf_data_mul.o		\
	src/matrix/tests/data/matrixf_data_qrdecomp.o		\
	src/matrix/tests/data/matrixf_data_transmul.o		\
								\
	src/matrix/tests/data/matrixcf_data_add.o		\
	src/matrix/tests/data/matrixcf_data_aug.o		\
	src/matrix/tests/data/matrixcf_data_chol.o		\
	src/matrix/tests/data/matrixcf_data_inv.o		\
	src/matrix/tests/data/matrixcf_data_linsolve.o		\
	src/matrix/tests/data/matrixcf_data_ludecomp.o		\
	src/matrix/tests/data/matrixcf_data_mul.o		\
	src/matrix/tests/data/matrixcf_data_qrdecomp.o		\
	src/matrix/tests/data/matrixcf_data_transmul.o		\

matrix_benchmarks :=						\
	src/matrix/bench/matrixf_inv_benchmark.c		\
	src/matrix/bench/matrixf_linsolve_benchmark.c		\
	src/matrix/bench/matrixf_mul_benchmark.c		\
	src/matrix/bench/smatrixf_mul_benchmark.c		\


# 
# MODULE : modem
#

modem_objects :=						\
	src/modem/src/ampmodem.o				\
	src/modem/src/gmskdem.o					\
	src/modem/src/gmskmod.o					\
	src/modem/src/modemf.o					\
	src/modem/src/modem_utilities.o				\
	src/modem/src/modem_apsk_const.o			\
	src/modem/src/modem_arb_const.o				\

# explicit targets and dependencies
modem_includes :=						\
	src/modem/src/freqdem.c					\
	src/modem/src/freqmod.c					\
	src/modem/src/modem_common.c				\
	src/modem/src/modem_psk.c				\
	src/modem/src/modem_dpsk.c				\
	src/modem/src/modem_ask.c				\
	src/modem/src/modem_qam.c				\
	src/modem/src/modem_apsk.c				\
	src/modem/src/modem_bpsk.c				\
	src/modem/src/modem_qpsk.c				\
	src/modem/src/modem_ook.c				\
	src/modem/src/modem_sqam32.c				\
	src/modem/src/modem_sqam128.c				\
	src/modem/src/modem_arb.c				\
	
#src/modem/src/modem_demod_soft_const.c

# main modem object
src/modem/src/modemf.o : %.o : %.c $(headers) $(modem_includes)

src/modem/src/gmskmod.o: %.o : %.c $(headers)

src/modem/src/gmskdem.o: %.o : %.c $(headers)

src/modem/src/ampmodem.o: %.o : %.c $(headers)

src/modem/src/freqmodem.o: %.o : %.c $(headers)

src/modem/src/modem_utilities.o: %.o : %.c $(headers)

src/modem/src/modem_apsk_const.o: %.o : %.c $(headers)

src/modem/src/modem_arb_const.o: %.o : %.c $(headers)


modem_autotests :=						\
	src/modem/tests/freqmodem_autotest.c			\
	src/modem/tests/modem_autotest.c			\
	src/modem/tests/modem_demodsoft_autotest.c		\
	src/modem/tests/modem_demodstats_autotest.c		\


modem_benchmarks :=						\
	src/modem/bench/gmskmodem_benchmark.c			\
	src/modem/bench/modem_modulate_benchmark.c		\
	src/modem/bench/modem_demodulate_benchmark.c		\
	src/modem/bench/modem_demodsoft_benchmark.c		\

# 
# MODULE : multichannel
#

multichannel_objects :=						\
	src/multichannel/src/firpfbch_crcf.o			\
	src/multichannel/src/firpfbch_cccf.o			\
	src/multichannel/src/ofdmframe.common.o			\
	src/multichannel/src/ofdmframegen.o			\
	src/multichannel/src/ofdmframesync.o			\

$(multichannel_objects) : %.o : %.c $(headers)

# list explicit targets and dependencies here
multichannel_includes :=					\
	src/multichannel/src/firpfbch.c				\
	src/multichannel/src/firpfbch2.c			\

src/multichannel/src/firpfbch_crcf.o : %.o : %.c $(headers) $(multichannel_includes)
src/multichannel/src/firpfbch_cccf.o : %.o : %.c $(headers) $(multichannel_includes)

# autotests
multichannel_autotests :=					\
	src/multichannel/tests/firpfbch2_crcf_autotest.c	\
	src/multichannel/tests/firpfbch_crcf_synthesizer_autotest.c	\
	src/multichannel/tests/firpfbch_crcf_analyzer_autotest.c	\
	src/multichannel/tests/ofdmframesync_autotest.c		\

# benchmarks
multichannel_benchmarks :=					\
	src/multichannel/bench/firpfbch_crcf_benchmark.c	\
	src/multichannel/bench/firpfbch2_crcf_benchmark.c	\
	src/multichannel/bench/ofdmframesync_acquire_benchmark.c	\
	src/multichannel/bench/ofdmframesync_rxsymbol_benchmark.c	\

# 
# MODULE : nco - numerically-controlled oscillator
#

nco_objects :=							\
	src/nco/src/nco_crcf.o					\
	src/nco/src/nco.utilities.o				\


src/nco/src/nco_crcf.o: %.o : %.c $(headers) src/nco/src/nco.c

src/nco/src/nco.utilities.o: %.o : %.c $(headers)


# autotests
nco_autotests :=						\
	src/nco/tests/nco_crcf_frequency_autotest.c		\
	src/nco/tests/nco_crcf_phase_autotest.c			\
	src/nco/tests/nco_crcf_pll_autotest.c			\
	src/nco/tests/unwrap_phase_autotest.c			\

# additional autotest objects
autotest_extra_obj +=						\
	src/nco/tests/data/nco_sincos_fsqrt1_2.o		\
	src/nco/tests/data/nco_sincos_fsqrt1_3.o		\
	src/nco/tests/data/nco_sincos_fsqrt1_5.o		\
	src/nco/tests/data/nco_sincos_fsqrt1_7.o		\

# benchmarks
nco_benchmarks :=						\
	src/nco/bench/nco_benchmark.c				\
	src/nco/bench/vco_benchmark.c				\

# 
# MODULE : optim - optimization
#

optim_objects :=						\
	src/optim/src/chromosome.o				\
	src/optim/src/gasearch.o				\
	src/optim/src/gradsearch.o				\
	src/optim/src/optim.common.o				\
	src/optim/src/qnsearch.o				\
	src/optim/src/utilities.o				\

$(optim_objects) : %.o : %.c $(headers)

# autotests
optim_autotests :=						\
	src/optim/tests/gradsearch_autotest.c			\

# benchmarks
optim_benchmarks :=


# 
# MODULE : quantization
#

quantization_objects :=						\
	src/quantization/src/compand.o				\
	src/quantization/src/quantizercf.o			\
	src/quantization/src/quantizerf.o			\
	src/quantization/src/quantizer.inline.o			\


src/quantization/src/compand.o: %.o : %.c $(headers)

src/quantization/src/quantizercf.o: %.o : %.c $(headers) src/quantization/src/quantizer.c

src/quantization/src/quantizerf.o: %.o : %.c $(headers) src/quantization/src/quantizer.c

src/quantization/src/quantizer.inline.o: %.o : %.c $(headers)


# autotests
quantization_autotests :=					\
	src/quantization/tests/compand_autotest.c		\
	src/quantization/tests/quantize_autotest.c		\


# benchmarks
quantization_benchmarks :=					\
	src/quantization/bench/quantizer_benchmark.c		\
	src/quantization/bench/compander_benchmark.c		\

# 
# MODULE : random
#

random_objects :=						\
	src/random/src/rand.o					\
	src/random/src/randn.o					\
	src/random/src/randexp.o				\
	src/random/src/randweib.o				\
	src/random/src/randgamma.o				\
	src/random/src/randnakm.o				\
	src/random/src/randricek.o				\
	src/random/src/scramble.o				\


$(random_objects) : %.o : %.c $(headers)

# autotests
random_autotests :=						\
	src/random/tests/scramble_autotest.c			\

#	src/random/tests/random_autotest.c


# benchmarks
random_benchmarks :=						\
	src/random/bench/random_benchmark.c			\


# 
# MODULE : sequence
#

sequence_objects :=						\
	src/sequence/src/bsequence.o				\
	src/sequence/src/msequence.o				\


$(sequence_objects) : %.o : %.c $(headers)


# autotests
sequence_autotests :=						\
	src/sequence/tests/bsequence_autotest.c			\
	src/sequence/tests/complementary_codes_autotest.c	\
	src/sequence/tests/msequence_autotest.c			\

# benchmarks
sequence_benchmarks :=						\
	src/sequence/bench/bsequence_benchmark.c		\

# 
# MODULE : utility
#

utility_objects :=						\
	src/utility/src/bshift_array.o				\
	src/utility/src/byte_utilities.o			\
	src/utility/src/msb_index.o				\
	src/utility/src/pack_bytes.o				\
	src/utility/src/shift_array.o				\

$(utility_objects) : %.o : %.c $(headers)

# autotests
utility_autotests :=						\
	src/utility/tests/bshift_array_autotest.c		\
	src/utility/tests/count_bits_autotest.c			\
	src/utility/tests/pack_bytes_autotest.c			\
	src/utility/tests/shift_array_autotest.c		\


# benchmarks
utility_benchmarks :=


# 
# MODULE : experimental
#

# legacy sources (not built, but keep around just in case)
#	src/experimental/legacy/dct.o
#	src/experimental/legacy/mdct.o
#	src/experimental/legacy/mdct_benchmark.c


# explicit targets and dependencies

experimental_filter_includes :=					\
	src/experimental/src/iirqmfb.c				\
	src/experimental/src/itqmfb.c				\
	src/experimental/src/qmfb.c				\
	src/experimental/src/symsync2.c				\
	src/experimental/src/symsynclp.c			\

src/experimental/src/filter_rrrf.o: %.o : %.c $(headers) $(experimental_filter_includes)

src/experimental/src/filter_crcf.o: %.o : %.c $(headers) $(experimental_filter_includes)

src/experimental/src/filter_cccf.o: %.o : %.c $(headers) $(experimental_filter_includes)




# Target collection
#
# Information about targets for each module is collected
# in these variables
objects :=							\
	src/libliquid.o						\
	$(agc_objects)						\
	$(audio_objects)					\
	$(buffer_objects)					\
	$(dotprod_objects)					\
	$(equalization_objects)					\
	$(fec_objects)						\
	$(fft_objects)						\
	$(filter_objects)					\
	$(framing_objects)					\
	$(math_objects)						\
	$(matrix_objects)					\
	$(modem_objects)					\
	$(multichannel_objects)					\
	$(nco_objects)						\
	$(optim_objects)					\
	$(quantization_objects)					\
	$(random_objects)					\
	$(sequence_objects)					\
	$(utility_objects)					\
	
	

autotest_sources :=						\
	autotest/null_autotest.c				\
	$(agc_autotests)					\
	$(audio_autotests)					\
	$(buffer_autotests)					\
	$(dotprod_autotests)					\
	$(equalization_autotests)				\
	$(fec_autotests)					\
	$(fft_autotests)					\
	$(filter_autotests)					\
	$(framing_autotests)					\
	$(math_autotests)					\
	$(matrix_autotests)					\
	$(modem_autotests)					\
	$(multichannel_autotests)				\
	$(nco_autotests)					\
	$(optim_autotests)					\
	$(quantization_autotests)				\
	$(random_autotests)					\
	$(sequence_autotests)					\
	$(utility_autotests)					\
	
	

benchmark_sources :=						\
	bench/null_benchmark.c					\
	$(agc_benchmarks)					\
	$(audio_benchmarks)					\
	$(buffer_benchmarks)					\
	$(dotprod_benchmarks)					\
	$(equalization_benchmarks)				\
	$(fec_benchmarks)					\
	$(fft_benchmarks)					\
	$(filter_benchmarks)					\
	$(framing_benchmarks)					\
	$(math_benchmarks)					\
	$(matrix_benchmarks)					\
	$(modem_benchmarks)					\
	$(multichannel_benchmarks)				\
	$(nco_benchmarks)					\
	$(optim_benchmarks)					\
	$(quantization_benchmarks)				\
	$(random_benchmarks)					\
	$(sequence_benchmarks)					\
	$(utility_benchmarks)					\
	


##
## TARGET : all       - build shared library (default)
##
.PHONY: all

# Shared library
SHARED_LIB	= libliquid.so

# liquid library definition
libliquid.a: $(objects)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@

# darwin
#
# gcc -dynamiclib -install_name libliquid.dylib -o libliquid.dylib libmodem.a libutility.a 
libliquid.dylib: $(objects)
	$(CC) -dynamiclib -install_name $@ -o $@ $^ $(LDFLAGS)

# linux, et al
libliquid.so: libliquid.a
	$(CC) -shared -Xlinker -soname=$@ -o $@ -Wl,-whole-archive $^ -Wl,-no-whole-archive $(LDFLAGS)

all: libliquid.a $(SHARED_LIB)

##
## TARGET : help      - print list of targets (see documentation for more)
##

# look for all occurences of '## TARGET : ' and print rest of line to screen
help:
	@echo "Targets for liquid-dsp makefile:"
	@$(GREP) -E "^## TARGET : " [Mm]akefile | $(SED) 's/## TARGET : /  /'

## 
## TARGET : install   - installs the libraries and header files in the host system
##

install:
	@echo "installing..."
	mkdir -p $(exec_prefix)/lib
	install -m 644 -p $(SHARED_LIB) libliquid.a $(exec_prefix)/lib
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/include/liquid
	install -m 644 -p $(addprefix include/,$(headers_install)) $(prefix)/include/liquid
	@echo ""
	@echo "---------------------------------------------------------"
	@echo "  liquid-dsp was successfully installed.     "
	@echo ""
	@echo "  On some machines (e.g. Linux) you should rebind your"
	@echo "  libraries by running 'ldconfig' to make the shared"
	@echo "  object available.  You might also need to modify your"
	@echo "  LD_LIBRARY_PATH environment variable to include the"
	@echo "  directory $(exec_prefix)"
	@echo ""
	@echo "  Please report bugs to $(BUGREPORT)"
	@echo "---------------------------------------------------------"
	@echo ""

## 
## TARGET : uninstall - uninstalls the libraries and header files in the host system
##

uninstall:
	@echo "uninstalling..."
	$(RM) $(addprefix $(prefix)/include/liquid/, $(headers_install))
	$(RM) $(exec_prefix)/lib/libliquid.a
	$(RM) $(exec_prefix)/lib/$(SHARED_LIB)
	@echo "done."

##
## autoscript
##

autoscript : scripts/autoscript

scripts/autoscript.o scripts/main.o : %.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

scripts/autoscript : scripts/autoscript.o scripts/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean-autoscript :
	$(RM) scripts/autoscript.o scripts/main.o scripts/autoscript


##
## TARGET : check     - build and run autotest scripts
##

# Autotests are used to check the validity and accuracy of the
# DSP libraries.

.PHONY: autotest
autotest_prog	= xautotest

# run the autotest generator script to create autotest_include.h
autotest_include.h : scripts/autoscript $(autotest_sources) $(include_headers)
	./scripts/autoscript $(PATHSEP) autotest $(autotest_sources) > $@

# autotest objects
# NOTE: by default, gcc compiles any file with a '.h' extension as a 'pre-compiled
#       header' so we need to explicity tell it to compile as a c source file with
#       the '-x c' flag
autotest_obj = $(patsubst %.c,%.o,$(autotest_sources))
$(autotest_obj) : %.o : %.c $(include_headers)
	$(CC) $(CFLAGS) $< -c -o $@

# additional autotest objects
$(autotest_extra_obj) : %.o : %.c $(include_headers)

# compile the autotest internal library functions without linking
autotest/autotestlib.o : autotest/autotestlib.c autotest/autotest.h
	$(CC) $(CFLAGS) $< -c -o $@

# compile the autotest program without linking
$(autotest_prog).o : autotest/autotest.c autotest/autotest.h autotest_include.h
	$(CC) $(CFLAGS) $< -c -o $@

# link the autotest program with the objects
# NOTE: linked libraries must come _after_ the target program
$(autotest_prog): $(autotest_prog).o $(autotest_obj) $(autotest_extra_obj) autotest/autotestlib.o libliquid.a
	$(CC) $^ -o $@ $(LDFLAGS)

# run the autotest program
check: $(autotest_prog)
	./$(autotest_prog) -v

# clean the generated files
clean-check:
	$(RM) autotest_include.h $(autotest_prog).o $(autotest_prog)
	$(RM) $(autotest_obj)
	$(RM) $(autotest_extra_obj)


##
## TARGET : bench     - build and run all benchmarks
##

# Benchmarks measure the relative speed of the DSP algorithms running
# on the target platform.
.PHONY: bench
bench_prog	= benchmark
BENCH_CFLAGS	= -Wall $(INCLUDE_CFLAGS) $(CONFIG_CFLAGS)
BENCH_LDFLAGS	= $(LDFLAGS)

# run the benchmark generator script to create benchmark_include.h
benchmark_include.h : scripts/autoscript $(benchmark_sources) $(include_headers)
	./scripts/autoscript $(PATHSEP) benchmark $(benchmark_sources) > $@

# benchmark objects
# NOTE: by default, gcc compiles any file with a '.h' extension as a 'pre-compiled
#       header' so we need to explicity tell it to compile as a c source file with
#       the '-x c' flag
benchmark_obj = $(patsubst %.c,%.o,$(benchmark_sources))
$(benchmark_obj) : %.o : %.c $(include_headers)
	$(CC) $(BENCH_CFLAGS) $< -c -o $@

# additional benchmark objects
$(benchmark_extra_obj) : %.o : %.c $(include_headers)

# compile the benchmark program without linking
$(bench_prog).o: bench/bench.c benchmark_include.h bench/bench.c
	$(CC) $(BENCH_CFLAGS) $< -c -o $(bench_prog).o

# link the benchmark program with the library objects
# NOTE: linked libraries must come _after_ the target program
$(bench_prog): $(bench_prog).o $(benchmark_obj) $(benchmark_extra_obj) libliquid.a
	$(CC) $^ -o $(bench_prog) $(BENCH_LDFLAGS)

# run the benchmark program
bench: $(bench_prog)
	./$(bench_prog)

# benchmark compare script
scripts/benchmark_compare : % : %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# fftbench program
bench/fftbench.o : %.o : %.c
	$(CC) $(BENCH_CFLAGS) $< -c -o $@

bench/fftbench : % : %.o libliquid.a
	$(CC) $^ -o $@ $(BENCH_LDFLAGS)

# clean up the generated files
clean-bench:
	$(RM) benchmark_include.h $(bench_prog).o $(bench_prog)
	$(RM) scripts/benchmark_compare
	$(RM) $(benchmark_obj)
	$(RM) $(benchmark_extra_obj)
	$(RM) bench/fftbench


## 
## TARGET : examples  - build all examples binaries
##
.PHONY: examples
example_programs :=						\
	examples/agc_crcf_example				\
	examples/agc_crcf_qpsk_example				\
	examples/agc_crcf_squelch_example			\
	examples/ampmodem_example				\
	examples/asgram_example					\
	examples/autocorr_cccf_example				\
	examples/bpacketsync_example				\
	examples/bpresync_example				\
	examples/bsequence_example				\
	examples/bufferf_example				\
	examples/chromosome_example				\
	examples/compand_example				\
	examples/compand_cf_example				\
	examples/complementary_codes_example			\
	examples/crc_example					\
	examples/cgsolve_example				\
	examples/cvsd_example					\
	examples/detector_cccf_example				\
	examples/dotprod_rrrf_example				\
	examples/dotprod_cccf_example				\
	examples/eqlms_cccf_blind_example			\
	examples/eqlms_cccf_example				\
	examples/eqrls_cccf_example				\
	examples/fec_example					\
	examples/fec_soft_example				\
	examples/fft_example					\
	examples/firdecim_crcf_example				\
	examples/firfarrow_rrrf_example				\
	examples/firfilt_rrrf_example				\
	examples/firdes_kaiser_example				\
	examples/firdespm_example				\
	examples/firhilb_example				\
	examples/firhilb_decim_example				\
	examples/firhilb_interp_example				\
	examples/firpfbch2_crcf_example				\
	examples/firinterp_crcf_example				\
	examples/firpfbch_crcf_example				\
	examples/firpfbch_crcf_analysis_example			\
	examples/firpfbch_crcf_synthesis_example		\
	examples/flexframesync_example				\
	examples/flexframesync_reconfig_example			\
	examples/framesync64_example				\
	examples/freqmodem_example				\
	examples/gasearch_example				\
	examples/gasearch_knapsack_example			\
	examples/gmskframesync_example				\
	examples/gmskmodem_example				\
	examples/gradsearch_datafit_example			\
	examples/gradsearch_example				\
	examples/interleaver_example				\
	examples/interleaver_soft_example			\
	examples/interleaver_scatterplot_example		\
	examples/iirdes_example					\
	examples/iirdes_analog_example				\
	examples/iirdes_pll_example				\
	examples/iirdecim_crcf_example				\
	examples/iirfilt_cccf_example				\
	examples/iirfilt_crcf_example				\
	examples/iirinterp_crcf_example				\
	examples/kbd_window_example				\
	examples/lpc_example					\
	examples/libliquid_example				\
	examples/matched_filter_example				\
	examples/math_lngamma_example				\
	examples/math_primitive_root_example			\
	examples/modem_arb_example				\
	examples/modem_example					\
	examples/modem_soft_example				\
	examples/modular_arithmetic_example			\
	examples/msequence_example				\
	examples/msresamp_crcf_example				\
	examples/nco_example					\
	examples/nco_pll_example				\
	examples/nco_pll_modem_example				\
	examples/nyquist_filter_example				\
	examples/ofdmflexframesync_example			\
	examples/ofdmframesync_example				\
	examples/packetizer_example				\
	examples/packetizer_soft_example			\
	examples/pll_example					\
	examples/polyfit_example				\
	examples/polyfit_lagrange_example			\
	examples/poly_findroots_example				\
	examples/quantize_example				\
	examples/qnsearch_example				\
	examples/random_histogram_example			\
	examples/repack_bytes_example				\
	examples/resamp_crcf_example				\
	examples/resamp2_crcf_example				\
	examples/resamp2_crcf_decim_example			\
	examples/resamp2_crcf_filter_example			\
	examples/resamp2_crcf_interp_example			\
	examples/scramble_example				\
	examples/smatrix_example				\
	examples/spgram_example					\
	examples/symsync_crcf_example				\
	examples/wdelayf_example				\
	examples/windowf_example				\
	examples/ricek_channel_example				\
	examples/linsolve_all_example              \
	

#	examples/metadata_example
#	examples/ofdmframegen_example
#	examples/gmskframe_example
#	examples/fading_generator_example

example_objects	= $(patsubst %,%.o,$(example_programs))
examples: $(example_programs)

EXAMPLES_LDFLAGS = $(LDFLAGS)

# NOTE: linked libraries must come _after_ the target program
$(example_objects): %.o : %.c

$(example_programs): % : %.o libliquid.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

#dct_compressor is special because we need tga
examples/dct_compressor: examples/dct_compressor.c tga/targa.c libliquid.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# clean examples
clean-examples:
	$(RM) examples/*.o
	$(RM) $(example_programs)

## 
## TARGET : sandbox   - build all sandbox binaries
##

# NOTE: sandbox _requires_ fftw3 to build
.PHONY: sandbox
sandbox_programs =						\
	sandbox/bpresync_test					\
	sandbox/cpmodem_test					\
	sandbox/count_ones_gentab				\
	sandbox/crc_gentab					\
	sandbox/ellip_func_test					\
	sandbox/ellip_test					\
	sandbox/eqlms_cccf_test					\
	sandbox/fecsoft_ber_test				\
	sandbox/fec_golay2412_test				\
	sandbox/fec_golay_test					\
	sandbox/fec_hamming3126_example				\
	sandbox/fec_hamming128_test				\
	sandbox/fec_hamming128_gentab				\
	sandbox/fec_hamming128_example				\
	sandbox/fec_hamming74_gentab				\
	sandbox/fec_hamming84_gentab				\
	sandbox/fec_hamming_test				\
	sandbox/fec_ldpc_test					\
	sandbox/fec_rep3_test					\
	sandbox/fec_rep5_test					\
	sandbox/fec_secded2216_test				\
	sandbox/fec_secded3932_test				\
	sandbox/fec_secded7264_test				\
	sandbox/fec_spc2216_test				\
	sandbox/fec_secded_punctured_test			\
	sandbox/fecsoft_conv_test				\
	sandbox/fecsoft_hamming128_gentab			\
	sandbox/fecsoft_ldpc_test				\
	sandbox/fec_sumproduct_test				\
	sandbox/fft_dual_radix_test				\
	sandbox/fft_mixed_radix_test				\
	sandbox/fft_recursive_plan_test				\
	sandbox/fft_recursive_test				\
	sandbox/fft_rader_prime_test				\
	sandbox/fft_rader_prime_radix2_test			\
	sandbox/fft_r2r_test					\
	sandbox/firdes_energy_test				\
	sandbox/firdes_fexp_test				\
	sandbox/firdes_gmskrx_test				\
	sandbox/firdes_length_test				\
	sandbox/firfarrow_rrrf_test				\
	sandbox/firpfbch_analysis_alignment_test		\
	sandbox/firpfbch2_analysis_equivalence_test		\
	sandbox/firpfbch2_test					\
	sandbox/firpfbch_analysis_equivalence_test		\
	sandbox/firpfbch_synthesis_equivalence_test		\
	sandbox/gmskmodem_test					\
	sandbox/householder_test				\
	sandbox/iirdes_example					\
	sandbox/iirfilt_intdiff_test				\
	sandbox/levinson_test					\
	sandbox/matched_filter_test				\
	sandbox/matched_filter_cfo_test				\
	sandbox/math_lngamma_test				\
	sandbox/math_cacosf_test				\
	sandbox/math_casinf_test				\
	sandbox/math_catanf_test				\
	sandbox/math_cexpf_test					\
	sandbox/math_clogf_test					\
	sandbox/math_csqrtf_test				\
	sandbox/matrix_test					\
	sandbox/minsearch_test					\
	sandbox/minsearch2_test					\
	sandbox/matrix_eig_test					\
	sandbox/modem_demodulate_arb_gentab			\
	sandbox/modem_demodulate_soft_test			\
	sandbox/modem_demodulate_soft_gentab			\
	sandbox/msresamp_crcf_test				\
	sandbox/ofdmoqam_firpfbch_test				\
	sandbox/ofdm_ber_test					\
	sandbox/ofdmframe_papr_test				\
	sandbox/ofdmframesync_cfo_test				\
	sandbox/packetizer_persistent_ber_test			\
	sandbox/pll_design_test					\
	sandbox/predemod_sync_test				\
	sandbox/quasinewton_test				\
	sandbox/resamp2_crcf_filterbank_test			\
	sandbox/resamp2_crcf_interp_recreate_test		\
	sandbox/reverse_byte_gentab				\
	sandbox/rkaiser2_test					\
	sandbox/simplex_test					\
	sandbox/symsync_crcf_test				\
	sandbox/symsync_eqlms_test				\
	sandbox/svd_test					\
	sandbox/thiran_allpass_iir_test				\

#	firpfbch_analysis_test
#	sandbox/ofdmoqam_firpfbch_cfo_test
#	sandbox/mdct_test
#	sandbox/fct_test


sandbox_objects	= $(patsubst %,%.o,$(sandbox_programs))
sandbox: $(sandbox_programs)
SANDBOX_LDFLAGS = $(LDFLAGS) -lfftw3f

# NOTE: linked libraries must come _after_ the target program
$(sandbox_objects): %.o : %.c

$(sandbox_programs): % : %.o libliquid.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# clean sandbox
clean-sandbox:
	$(RM) sandbox/*.o
	$(RM) $(sandbox_programs)


## 
## TARGET : doc       - build documentation (doc/liquid.pdf)
##

doc : doc/liquid.pdf

doc/liquid.pdf : libliquid.a
	cd doc ; make

clean-doc :
	cd doc ; make mostlyclean

distclean-doc :
	cd doc ; make distclean


# Clean
.PHONY: clean

##
## TARGET : world     - build absolutely everything
##
world : all check bench examples sandbox doc

##
## TARGET : clean     - clean build (objects, dependencies, libraries, etc.)
##

clean-modules:
	@echo "cleaning modules..."
	$(RM) src/agc/src/*.o          src/agc/bench/*.o          src/agc/tests/*.o
	$(RM) src/audio/src/*.o        src/audio/bench/*.o        src/audio/tests/*.o
	$(RM) src/buffer/src/*.o       src/buffer/bench/*.o       src/buffer/tests/*.o
	$(RM) src/dotprod/src/*.o      src/dotprod/bench/*.o      src/dotprod/tests/*.o
	$(RM) src/equalization/src/*.o src/equalization/bench/*.o src/equalization/tests/*.o
	$(RM) src/fec/src/*.o          src/fec/bench/*.o          src/fec/tests/*.o
	$(RM) src/fft/src/*.o          src/fft/bench/*.o          src/fft/tests/*.o
	$(RM) src/filter/src/*.o       src/filter/bench/*.o       src/filter/tests/*.o
	$(RM) src/framing/src/*.o      src/framing/bench/*.o      src/framing/tests/*.o
	$(RM) src/math/src/*.o         src/math/bench/*.o         src/math/tests/*.o
	$(RM) src/matrix/src/*.o       src/matrix/bench/*.o       src/matrix/tests/*.o
	$(RM) src/modem/src/*.o        src/modem/bench/*.o        src/modem/tests/*.o
	$(RM) src/multichannel/src/*.o src/multichannel/bench/*.o src/multichannel/tests/*.o
	$(RM) src/nco/src/*.o          src/nco/bench/*.o          src/nco/tests/*.o
	$(RM) src/optim/src/*.o        src/optim/bench/*.o        src/optim/tests/*.o
	$(RM) src/quantization/src/*.o src/quantization/bench/*.o src/quantization/tests/*.o
	$(RM) src/random/src/*.o       src/random/bench/*.o       src/random/tests/*.o
	$(RM) src/utility/src/*.o      src/utility/bench/*.o      src/utility/tests/*.o
	$(RM) src/experimental/src/*.o src/experimental/bench/*.o src/experimental/tests/*.o
	$(RM) libliquid.o

clean: clean-modules clean-autoscript clean-check clean-bench clean-examples clean-sandbox clean-doc
	$(RM) $(extra_clean)
	$(RM) libliquid.a
	$(RM) $(SHARED_LIB)

##
## TARGET : distclean - removes everything except the originally distributed files
##

distclean: clean distclean-doc
	@echo "cleaning distribution..."
	$(RM) octave-core *.m
	$(RM) configure config.h config.h.in config.h.in~ config.log config.status
	$(RM) -r autom4te.cache
	$(RM) makefile

