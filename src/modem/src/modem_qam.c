/*
 * Copyright (c) 2007, 2008, 2009, 2010, 2011, 2012, 2013 Joseph Gaeddert
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
// modem_qam.c
//

#include <assert.h>

// create a qam (quaternary amplitude-shift keying) modem object
MODEM() MODEM(_create_qam)(unsigned int _bits_per_symbol)
{
    if (_bits_per_symbol < 1 ) {
        fprintf(stderr,"error: modem_create_qam(), modem must have at least 2 bits/symbol\n");
        exit(1);
    }

    MODEM() q = (MODEM()) malloc( sizeof(struct MODEM(_s)) );

    MODEM(_init)(q, _bits_per_symbol);

    if (q->m % 2) {
        // rectangular qam
        q->data.qam.m_i = (q->m + 1) >> 1;
        q->data.qam.m_q = (q->m - 1) >> 1;
    } else {
        // square qam
        q->data.qam.m_i = q->m >> 1;
        q->data.qam.m_q = q->m >> 1;
    }

    q->data.qam.M_i = 1 << (q->data.qam.m_i);
    q->data.qam.M_q = 1 << (q->data.qam.m_q);

    assert(q->data.qam.m_i + q->data.qam.m_q == q->m);
    assert(q->data.qam.M_i * q->data.qam.M_q == q->M);

    switch (q->M) {
    case 4:    q->data.qam.alpha = RQAM4_ALPHA;    q->scheme = LIQUID_MODEM_QAM4;   break;
    case 8:    q->data.qam.alpha = RQAM8_ALPHA;    q->scheme = LIQUID_MODEM_QAM8;   break;
    case 16:   q->data.qam.alpha = RQAM16_ALPHA;   q->scheme = LIQUID_MODEM_QAM16;  break;
    case 32:   q->data.qam.alpha = RQAM32_ALPHA;   q->scheme = LIQUID_MODEM_QAM32;  break;
    case 64:   q->data.qam.alpha = RQAM64_ALPHA;   q->scheme = LIQUID_MODEM_QAM64;  break;
    case 128:  q->data.qam.alpha = RQAM128_ALPHA;  q->scheme = LIQUID_MODEM_QAM128; break;
    case 256:  q->data.qam.alpha = RQAM256_ALPHA;  q->scheme = LIQUID_MODEM_QAM256; break;
#if 0
    case 512:  q->data.qam.alpha = RQAM512_ALPHA;     break;
    case 1024: q->data.qam.alpha = RQAM1024_ALPHA;    break;
    case 2048: q->data.qam.alpha = RQAM2048_ALPHA;    break;
    case 4096: q->data.qam.alpha = RQAM4096_ALPHA;    break;
    default:
        // calculate alpha dynamically
        // NOTE: this is only an approximation
        q->data.qam.alpha = sqrtf(2.0f / (T)(q->M) );
#else
    default:
        fprintf(stderr,"error: modem_create_qam(), cannot support QAM with m > 8\n");
        exit(1);
#endif
    }

    unsigned int k;
    for (k=0; k<(q->m); k++)
        q->ref[k] = (1<<k) * q->data.qam.alpha;

    q->modulate_func = &MODEM(_modulate_qam);
    q->demodulate_func = &MODEM(_demodulate_qam);

    // initialize symbol map
    q->symbol_map = (TC*)malloc(q->M*sizeof(TC));
    MODEM(_init_map)(q);
    q->modulate_using_map = 1;

    // initialize soft-demodulation look-up table
    if      (q->m == 3) MODEM(_demodsoft_gentab)(q, 3);
    else if (q->m >= 4) MODEM(_demodsoft_gentab)(q, 4);

    // reset and return
    MODEM(_reset)(q);
    return q;
}

// modulate QAM
void MODEM(_modulate_qam)(MODEM()      _q,
                          unsigned int _sym_in,
                          TC *         _y)
{
    unsigned int s_i;   // in-phase symbol
    unsigned int s_q;   // quadrature symbol
    s_i = _sym_in >> _q->data.qam.m_q;
    s_q = _sym_in & ( (1<<_q->data.qam.m_q)-1 );

    // 'encode' symbols (actually gray decoding)
    s_i = gray_decode(s_i);
    s_q = gray_decode(s_q);

    // compute output sample
    *_y = (2*(int)s_i - (int)(_q->data.qam.M_i) + 1) * _q->data.qam.alpha +
          (2*(int)s_q - (int)(_q->data.qam.M_q) + 1) * _q->data.qam.alpha * _Complex_I;
}

// demodulate QAM
void MODEM(_demodulate_qam)(MODEM()      _q,
                          TC             _x,
                          unsigned int * _sym_out)
{
    // demodulate in-phase component on linearly-spaced array
    unsigned int s_i;   // in-phase symbol
    T res_i;        // in-phase residual
    MODEM(_demodulate_linear_array_ref)(crealf(_x), _q->data.qam.m_i, _q->ref, &s_i, &res_i);

    // demodulate quadrature component on linearly-spaced array
    unsigned int s_q;   // quadrature symbol
    T res_q;        // quadrature residual
    MODEM(_demodulate_linear_array_ref)(cimagf(_x), _q->data.qam.m_q, _q->ref, &s_q, &res_q);

    // 'decode' output symbol (actually gray encoding)
    s_i = gray_encode(s_i);
    s_q = gray_encode(s_q);
    *_sym_out = ( s_i << _q->data.qam.m_q ) + s_q;

    // re-modulate symbol (subtract residual) and store state
    _q->x_hat = _x - (res_i + _Complex_I*res_q);
    _q->r = _x;
}

