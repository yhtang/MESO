#ifndef LMP_MESO_SPH_KERNEL
#define LMP_MESO_SPH_KERNEL

#include <cmath>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "math_constants.h"

#include "util_meso.h"


namespace LAMMPS_NS
{

template<typename REAL>
struct SPHKernelTang3D {
    inline __device__ __host__ SPHKernelTang3D( REAL rc_inv_ ) : rc_inv( rc_inv_ ) {}

    inline __device__ __host__ REAL operator()( REAL r )
    {
        REAL s = REAL( 1 ) - r * rc_inv;
        REAL s2 = s * s;
        REAL s4 = s2 * s2;
        return s4 * ( REAL( 6.2885662450433414961 ) - REAL( 6.8602540867169121697 ) * s2 + REAL( 2.0009074418437293730 ) * s4 );
    }
    inline __device__ __host__ REAL gradient( REAL r )
    {
        REAL s = REAL( 1 ) - r * rc_inv;
        REAL s2 = s * s;
        REAL s4 = s2 * s2;
        return rc_inv * s * s2 * ( REAL( -25.154264980173365984 ) + REAL( 41.161524520301469465 ) * s2 - REAL( 16.007259534749834984 ) * s4 );
    }
    inline static __device__ __host__ REAL norm( REAL rc )
    {
        return REAL( 0.46131289592249569154 ) * power<3>( rc );
    }

    const REAL rc_inv;
};

const static double _PI      = 3.1415926535897932385;
const static double _PI_SQRT = 1.77245385090551602730;

enum SPHKernelType {
    KERNEL_GAUSSIAN  = 0,
    KERNEL_LUCY      = 1,
    KERNEL_CUBIC     = 2,
    KERNEL_QUARTIC   = 3,
    KERNEL_QUINTIC   = 4
};

#ifdef __CUDACC__

template<typename REAL> struct SPHKernelParam {
    __device__ __host__ SPHKernelParam( REAL rc_, REAL hinv_, REAL sigma_, REAL coef_ ) :
        pi( _PI ),
        pi_sqrt( _PI_SQRT ),
        rc( rc_ ),
        hinv( hinv_ ),
        sigma( sigma_ ),
        coef( coef_ ),
        coef_g( coef*hinv ),
        kappa( rc*hinv ),
        erf_kappa( erf( kappa ) ),
        exp_kappa2( exp( -kappa*kappa ) ),
        kib_coef( sigma * pi )
    {
    }

    const REAL pi, pi_sqrt;
    const REAL rc, hinv, kappa;
    const REAL sigma;
    const REAL coef, coef_g;
    const REAL erf_kappa, exp_kappa2;
    const REAL kib_coef;
};

template<typename REAL>
struct SPHKernelGauss1D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelGauss1D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 3.0 / rc_, 1.0 / _PI_SQRT, 1.0 / _PI_SQRT * 3.0 / rc_ ) {}

    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return exp( -s * s ) * this->coef;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return -2.0 * s * exp( -s * s ) * this->coef_g;
    }
};

template<typename REAL>
struct SPHKernelGauss2D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelGauss2D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 3.0 / rc_, 1.0 / _PI, 1.0 / _PI * power<2>( 3.0 / rc_ ) ) {}

    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return exp( -s * s ) * this->coef;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return -2.0 * s * exp( -s * s ) * this->coef_g;
    }
};

template<typename REAL>
struct SPHKernelGauss3D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelGauss3D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 3.0 / rc_, 1.0 / _PI / _PI_SQRT, 1.0 / _PI / _PI_SQRT * power<3>( 3.0 / rc_ ) ) {}

    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return exp( -s * s ) * this->coef;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return -2.0 * s * exp( -s * s ) * this->coef_g;
    }
};

template<typename REAL>
struct SPHKernelLucy1D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelLucy1D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 1.0 / rc_, 5.0 / 4.0, 5.0 / 4.0 * 1.0 / rc_ ),
        zero( 0.0 ), one( 1.0 ), three( 3.0 ), _twelve( -12.0 )
    {
    }

    const REAL zero, one, three, _twelve;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? ( one + three * s ) * power<3>( one - s ) * this->coef : zero;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? _twelve * s * power<2>( one - s ) * this->coef_g : zero;
    }
};

template<typename REAL>
struct SPHKernelLucy2D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelLucy2D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 1.0 / rc_, 5.0 / _PI, 5.0 / _PI * power<2>( 1.0 / rc_ ) ),
        zero( 0.0 ), one( 1.0 ), three( 3.0 ), _twelve( -12.0 )
    {
    }

    const REAL zero, one, three, _twelve;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? ( one + three * s ) * power<3>( one - s ) * this->coef : zero;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? _twelve * s * power<2>( one - s ) * this->coef_g : zero;
    }
};

template<typename REAL>
struct SPHKernelLucy3D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelLucy3D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 1.0 / rc_, 105.0 / 16.0 / _PI, 105.0 / 16.0 / _PI * power<3>( 1.0 / rc_ ) ),
        zero( 0.0 ), one( 1.0 ), three( 3.0 ), _twelve( -12.0 )
    {
    }

    const REAL zero, one, three, _twelve;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? ( one + three * s ) * power<3>( one - s ) * this->coef : zero;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        return s < one ? _twelve * s * power<2>( one - s ) * this->coef_g : zero;
    }
};

template<typename REAL>
struct SPHKernelCubic1D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelCubic1D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 2.0 / rc_, 2.0 / 3.0, 2.0 / 3.0 * 2.0 / rc_ ),
        zero( 0.0 ), quarter( 0.25 ), half( 0.5 ), one( 1.0 ), two( 2.0 ), three( 3.0 )
    {
    }

    const REAL zero, quarter, half, one, two, three;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w;
        if( s < 1.0 ) {
            w = one - three * power<2>( s ) * half + three * power<3>( s ) * quarter;
        } else if( s < 2.0 ) {
            w = power<3>( two - s ) * quarter;
        } else {
            w = zero;
        }
        w *= this->coef;
        return w;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w_g;
        if( s < 1.0 ) {
            w_g = -three * s + power<2>( three ) * s * s * quarter;
        } else if( s < 2.0 ) {
            w_g = -three * power<2>( 2 - s ) * quarter;
        } else {
            w_g = 0.0;
        }
        w_g *= this->coef_g;
        return w_g;
    }
};

template<typename REAL>
struct SPHKernelCubic2D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelCubic2D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 2.0 / rc_, 10.0 / 7.0 / _PI, 10.0 / 7.0 / _PI * power<2>( 2.0 / rc_ ) ),
        zero( 0.0 ), quarter( 0.25 ), half( 0.5 ), one( 1.0 ), two( 2.0 ), three( 3.0 )
    {
    }

    const REAL zero, quarter, half, one, two, three;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w;
        if( s < 1.0 ) {
            w = one - three * power<2>( s ) * half + three * power<3>( s ) * quarter;
        } else if( s < 2.0 ) {
            w = power<3>( two - s ) * quarter;
        } else {
            w = zero;
        }
        w *= this->coef;
        return w;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w_g;
        if( s < 1.0 ) {
            w_g = -three * s + power<2>( three ) * s * s * quarter;
        } else if( s < 2.0 ) {
            w_g = -three * power<2>( 2 - s ) * quarter;
        } else {
            w_g = 0.0;
        }
        w_g *= this->coef_g;
        return w_g;
    }
};

template<typename REAL>
struct SPHKernelCubic3D: public SPHKernelParam<REAL> {
    __device__ __host__ SPHKernelCubic3D( REAL rc_ ) :
        SPHKernelParam<REAL>( rc_, 2.0 / rc_, 1.0 / _PI, 1.0 / _PI * power<3>( 2.0 / rc_ ) ),
        zero( 0.0 ), quarter( 0.25 ), half( 0.5 ), one( 1.0 ), two( 2.0 ), three( 3.0 )
    {
    }

    const REAL zero, quarter, half, one, two, three;
    inline __device__ __host__ REAL operator()( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w;
        if( s < 1.0 ) {
            w = one - three * power<2>( s ) * half + three * power<3>( s ) * quarter;
        } else if( s < 2.0 ) {
            w = power<3>( two - s ) * quarter;
        } else {
            w = zero;
        }
        w *= this->coef;
        return w;
    }
    inline __device__ __host__ REAL gradient( REAL r ) const
    {
        REAL s = r * this->hinv;
        REAL w_g;
        if( s < 1.0 ) {
            w_g = -three * s + power<2>( three ) * s * s * quarter;
        } else if( s < 2.0 ) {
            w_g = -three * power<2>( 2 - s ) * quarter;
        } else {
            w_g = 0.0;
        }
        w_g *= this->coef_g;
        return w_g;
    }
};

#endif

}

#endif
