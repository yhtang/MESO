#ifndef LMP_MESO_MATH
#define LMP_MESO_MATH

#include <cmath>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "math_constants.h"

#include "vec_op_meso.h"
#include "util_meso.h"

#ifndef EPSILON
#define EPSILON         1.0E-10
#define EPSILON_SQ      1.0E-20
#endif
#define _LN_2           6.9314718055994528623E-1
#define _LOG2_E         1.4426950408889634074
#define _1_OVER_SQ2     7.0710678118654757274E-1
#define _SQRT_2         1.4142135623730950488
#define _SQRT_3         1.7320508075688772935
#define _2_TO_MINUS_30  9.3132257461547851562E-10
#define _2_TO_MINUS_31  4.6566128730773925781E-10
#define _2_TO_MINUS_32  2.3283064365386962891E-10

#if(__CUDA_ARCH__<=530)
#define WARPSZ      32U
#define WARPALIGN (~31U)
#else
#error UNKNOWN ARCHITECTURE FOR DETERMINING WARP SIZE
#endif

inline int ceiling( int x, int inc )
{
    return ( ( x + inc - 1 ) / inc ) * inc ;
}

inline unsigned int n_block( unsigned int threads, unsigned int threadPerBlock )
{
    return ( threads + threadPerBlock - 1 ) / threadPerBlock;
}

inline unsigned int popc( unsigned int v )
{
    unsigned int c;
    for( c = 0; v; v >>= 1 ) c += v & 1;
    return c;
}


#ifdef __CUDACC__

template<typename REAL>
__inline__ __device__ __host__ REAL polyval( REAL x, int order, REAL const *a ) {
	REAL r = *a++;
	for( int i = 0; i < order; i++ ) r = r * x + *a++;
	return r;
}

// evaluate the integral of the given polynomial
template<typename REAL>
__inline__ __device__ __host__ REAL polyval_integral( REAL x, int order, REAL const *a ) {
	REAL r = (*a++) / (order+1);
	for( int i = 0; i < order; i++ ) r = r * x + (*a++) / (order-i);
	return r * x;
}

__inline__ __device__ uint __float_as_uint( float r )
{
    uint u = 0;
    asm volatile( "mov.b32 %0, %1;" : "=r"( u ) : "f"( r ) );
    return u;
}

__inline__ __device__ float __uint_as_float( uint u )
{
    float r;
    asm volatile( "mov.b32 %0, %1;" : "=f"( r ) : "r"( u ) );
    return r;
}

__inline__ __device__ double __int_as_double( int lo )
{
    double r;
    int hi = 0;
    asm volatile( "mov.b64 %0,{%1,%2};" : "=d"( r ) : "r"( lo ), "r"( hi ) );
    return r;
}

__inline__ __device__ int __double_as_int( double r )
{
    __attribute__( ( unused ) ) int hi, lo;
    asm volatile( "mov.b64 {%0,%1},%2;" : "=r"( lo ), "=r"( hi ) : "d"( r ) );
    return lo;
}

__inline__ __device__ void __double2hiloint( double x, int &hi, int &lo )
{
    asm volatile( "mov.b64 {%0, %1}, %2;" : "=r"( lo ), "=r"( hi ) : "d"( x ) );
}

__inline__ __device__ uint __double2louint( double x )
{
    __attribute__( ( unused ) ) uint lo, hi;
    asm volatile( "mov.b64 {%0, %1}, %2;" : "=r"( lo ), "=r"( hi ) : "d"( x ) );
    return lo;
}

__inline__ __device__ uint __vabsdiff4( uint u, uint v )
{
    uint w;
    asm volatile( "vabsdiff4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"( w ) : "r"( u ), "r"( v ), "r"( w ) );
    return w;
}

template<uint N> __inline__ __device__ __host__ double power( const double x )
{
    return power<2>( power < N / 2 > ( x ) ) * power < N % 2 > ( x );
}

template<> __inline__ __device__ __host__ double power<2>( const double x )
{
    return x * x;
}

template<> __inline__ __device__ __host__ double power<1>( const double x )
{
    return x;
}

template<> __inline__ __device__ __host__ double power<0>( __attribute__( ( unused ) ) const double x )
{
    return 1.0;
}

__inline__ __device__ __host__ void pred_swap( bool pred, int &u, int &v )
{
    int I = pred ? u : v;
    int J = pred ? v : u;
    u = I, v = J;
}

__inline__ __host__ __device__ int calc_bid( int bidX, int bidY, int bidZ, int bnX, int bnY )
{
    return bidX + bnX * ( bidY + bidZ * bnY );
}

__inline__ __host__ __device__ double minimum_image( double dr, double p )
{
    double p_half = p * 0.5;
    return dr + ( dr > -p_half ? ( dr < p_half ? 0.0 : -p ) : p );
}

// clamp at [nmin, nmax)
__inline__ __host__ __device__ int clamp( int i, int nmin, int nmax )
{
    return max( nmin, min( i, nmax - 1 ) );
}

// clamp at [min, max]
template<typename T>
__inline__ __host__ __device__ T bound( T x, T lower, T upper )
{
    return max( lower, min( x, upper ) );
}
__inline__ __host__ __device__ uint bit_space3( uint x )
{
    x = ( x | ( x << 12 ) ) & 0X00FC003FU;
    x = ( x | ( x <<  6 ) ) & 0X381C0E07U;
    x = ( x | ( x <<  4 ) ) & 0X190C8643U;
    x = ( x | ( x <<  2 ) ) & 0X49249249U;
    return x;
}

__inline__ __host__ __device__ uint interleave3( uint i, uint j, uint k )
{
    return bit_space3( i ) | ( bit_space3( j ) << 1 ) | ( bit_space3( k ) << 2 ) ;
}

__inline__ __device__ __host__ uint morton_encode( uint x, uint y, uint z )
{
    return interleave3( x, y, z );
}

__inline__ __device__ __host__ void morton_decode( uint code, uint &x, uint &y, uint &z )
{
    x = code & 0x09249249;            // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = ( x ^ ( x >>  2 ) ) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = ( x ^ ( x >>  4 ) ) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = ( x ^ ( x >>  8 ) ) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = ( x ^ ( x >> 16 ) ) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    y = ( code >> 1 ) & 0x09249249;
    y = ( y ^ ( y >>  2 ) ) & 0x030c30c3;
    y = ( y ^ ( y >>  4 ) ) & 0x0300f00f;
    y = ( y ^ ( y >>  8 ) ) & 0xff0000ff;
    y = ( y ^ ( y >> 16 ) ) & 0x000003ff;
    z = ( code >> 2 ) & 0x09249249;
    z = ( z ^ ( z >>  2 ) ) & 0x030c30c3;
    z = ( z ^ ( z >>  4 ) ) & 0x0300f00f;
    z = ( z ^ ( z >>  8 ) ) & 0xff0000ff;
    z = ( z ^ ( z >> 16 ) ) & 0x000003ff;
}

__inline__ __device__ double __2_to_n( int n )
{
    return __hiloint2double( ( 1023 + n ) << 20, 0 );
}

// error < 2e-16
__inline__ __device__ double __rsqrt( double x )
{
    double xrsqrt = __longlong_as_double( 0X5FE660FCB5422422LL - ( __double_as_longlong( x ) >> 1 ) );

    // iterative, quadratic convergence
    double x2m = x * -0.5;
    xrsqrt  *= __fma_rn( xrsqrt * xrsqrt, x2m, 1.5 );
    xrsqrt  *= __fma_rn( xrsqrt * xrsqrt, x2m, 1.5 );
    xrsqrt  *= __fma_rn( xrsqrt * xrsqrt, x2m, 1.5 );
    xrsqrt  *= __fma_rn( xrsqrt * xrsqrt, x2m, 1.5 );
    return xrsqrt;
}

__inline__ __device__ double __sqrtd( double x )
{
    return x * __rsqrt( x );
}

// a \in [1e-208,1e208] otherwise accuracy is not guaranteed
// get IEEE-double-precision accurate result within 4 iterations, error <= 1ULP
__inline__ __device__ double __rcp( double x )
{
    double xinv = __longlong_as_double( 0X7FDE62361B1C4042LL - __double_as_longlong( x ) );
    xinv -= __fma_rn( x, xinv, -1. ) * xinv;
    xinv -= __fma_rn( x, xinv, -1. ) * xinv;
    xinv -= __fma_rn( x, xinv, -1. ) * xinv;
    xinv -= __fma_rn( x, xinv, -1. ) * xinv;
    return xinv;
}

// the code below is equivalent to the above lines, but is optimized to fully use only FMA, and is a lot harder to read
__inline__ __device__ double __rcp_fma( double x )
{
    double xinv = __longlong_as_double( 0X7FDE62361B1C4042LL - __double_as_longlong( x ) );

    double b;
    b = __fma_rn( __fma_rn( x, xinv, -1. ), xinv, -xinv );
    xinv = __fma_rn( __fma_rn( b, x, 1. ), -b, -b );
    b = __fma_rn( __fma_rn( x, xinv, -1. ), xinv, -xinv );
    xinv = __fma_rn( __fma_rn( b, x, 1. ), -b, -b );
    return xinv;
}

// x \in [1,2]
// 14-order fast converging Chebyshev polynomial, log2(x)=z*P(z^2), error < 1ULP
#define __LOG2D_0  2.8853900817779268114E+0
#define __LOG2D_1  2.8312651192953993354E-2
#define __LOG2D_2  5.0006798065881969549E-4
#define __LOG2D_3  1.0514733588011180538E-5
#define __LOG2D_4  2.4074128088151586443E-7
#define __LOG2D_5  5.7988453014506741861E-9
#define __LOG2D_6  1.4374842194796670219E-10
#define __LOG2D_7  4.0928048937567843469E-12

__inline__  __device__ double __log2d_frac( double x )
{
    bool pred = x > _SQRT_2;
    x *= pred ? 0.5 : _1_OVER_SQ2 ;
    double z = ( x - 1. ) * __rcp( x + 1. );
    double y = z * z * 33.9705627484771406;
    double s = __LOG2D_7;
    s = __fma_rn( s, y, __LOG2D_6 );
    s = __fma_rn( s, y, __LOG2D_5 );
    s = __fma_rn( s, y, __LOG2D_4 );
    s = __fma_rn( s, y, __LOG2D_3 );
    s = __fma_rn( s, y, __LOG2D_2 );
    s = __fma_rn( s, y, __LOG2D_1 );
    s = __fma_rn( s, y, __LOG2D_0 );
    return __fma_rn( z, s, ( pred ? 1.0 : 0.5 ) );
}

// a \in (0,+Infty]
__inline__ __device__ double __log2d( double a )
{
    int hi, lo;
    __double2hiloint( a, hi, lo );
    // extract the exponent
    int I = ( hi >> 20 ) - 1023;
    // reset the exponent to 0 and do fractional log
    double F = __log2d_frac( __hiloint2double( ( hi & 0X000FFFFF ) | 0X3FF00000, lo ) );
    return I + F ;
}

// x \in [0,1.0]
// 11th order 2^x = P(x), error < 1 ULP
#define __EXP2_0   9.9999999999999999572E-1
#define __EXP2_1   6.9314718055994653980E-1
#define __EXP2_2   2.4022650695904222220E-1
#define __EXP2_3   5.5504108665909870679E-2
#define __EXP2_4   9.6181290971755593396E-3
#define __EXP2_5   1.3333558738165095559E-3
#define __EXP2_6   1.5403509189194102748E-4
#define __EXP2_7   1.5253232908458899497E-5
#define __EXP2_8   1.3207676270599404858E-6
#define __EXP2_9   1.0258347084283025531E-7
#define __EXP2_10  6.5379419072372670333E-9
#define __EXP2_11  6.3026908837748924689E-10

__inline__ __device__ double __exp2d_frac( double x )
{
    double s = __EXP2_11;
    s = __fma_rn( s, x, __EXP2_10 );
    s = __fma_rn( s, x, __EXP2_9 );
    s = __fma_rn( s, x, __EXP2_8 );
    s = __fma_rn( s, x, __EXP2_7 );
    s = __fma_rn( s, x, __EXP2_6 );
    s = __fma_rn( s, x, __EXP2_5 );
    s = __fma_rn( s, x, __EXP2_4 );
    s = __fma_rn( s, x, __EXP2_3 );
    s = __fma_rn( s, x, __EXP2_2 );
    s = __fma_rn( s, x, __EXP2_1 );
    s = __fma_rn( s, x, __EXP2_0 );
    return s;
}

__inline__ __device__ double __exp2d( double a )
{
    double I = floor( a ) ;
    return __2_to_n( I ) * __exp2d_frac( a - I );
}

// for a \in [1E-102,1E102], b \in [0,3], relative error < 7.85E-16
__inline __device__ double __powd( double a, double b )
{
    int hi, lo;
    __double2hiloint( a, hi, lo );
    // extract the exponent
    double I  = ( hi >> 20 ) - 1023;
    // reset the exponent to 0 and do fractional log
    double F  = __log2d_frac( __hiloint2double( ( hi & 0X000FFFFF ) | 0X3FF00000, lo ) );
    // multiply by the exponent, separate into two parts to avoid losing mantissa precision
    double II = floor( b * ( I + F ) );
    return __2_to_n( II ) * __exp2d_frac( __fma_rn( b, F, __fma_rn( b, I, -II ) ) );
//  return __exp2d( b * __log2d(a) ); accuracy degrade when exponents gets large
}

// modified sinpi to take in x \in [-1.0,1.0]
// 12-th order Chebyshev polynomial, maximum absolute error 7.5*10^-13
// that is enough because the underlying random number is only 31bit = 9.33 decimal digit
#define _SINPI_0   9.99999999999249900E-1
#define _SINPI_1  -1.23370055006260170E+0
#define _SINPI_2   2.53669506722714547E-1
#define _SINPI_3  -2.08634736828917670E-2
#define _SINPI_4   9.19240000031795430E-4
#define _SINPI_5  -2.51721958850906157E-5
#define _SINPI_6   4.49220128554338954E-7

__inline__  __device__ double __sinpi( double x )
{
    x  = 2.0 * x - 1.0;
    x *= x;
    double s = _SINPI_6;
    s = __fma_rn( s, x, _SINPI_5 );
    s = __fma_rn( s, x, _SINPI_4 );
    s = __fma_rn( s, x, _SINPI_3 );
    s = __fma_rn( s, x, _SINPI_2 );
    s = __fma_rn( s, x, _SINPI_1 );
    s = __fma_rn( s, x, _SINPI_0 );
    return s;
}

// modified cospi to take in x \in [-1.0,1.0]
// 11-th order Chebyshev polynomial, maximum relative error 1.104*10^-10, 33.08 bits accurate
#define _COSPI_0  -1.57079632662144460E+0
#define _COSPI_1   6.45964092644060746E-1
#define _COSPI_2  -7.96925872866600517E-2
#define _COSPI_3   4.68162024021793872E-3
#define _COSPI_4  -1.60217135750921262E-4
#define _COSPI_5   3.41817283473266926E-6

__inline__  __device__ double __cospi( double x )
{
    x  = 2.0 * x - 1.0;
    double x2 = x * x;
    double s  = _COSPI_5;
    s = __fma_rn( s, x2, _COSPI_4 );
    s = __fma_rn( s, x2, _COSPI_3 );
    s = __fma_rn( s, x2, _COSPI_2 );
    s = __fma_rn( s, x2, _COSPI_1 );
    s = __fma_rn( s, x2, _COSPI_0 );
    return s * x;
}

// x \in [1,2]
// 8th order fast convergence Chebyshev polynomial, maximal relative error 4.21E-12, accurate bits = 37.8
#define _LOG2U_0  2.88539008179006374E+0
#define _LOG2U_1  2.83126505877817866E-2
#define _LOG2U_2  5.00072802051539862E-4
#define _LOG2U_3  1.05013262724846015E-5
#define _LOG2U_4  2.55854634203511155E-7

__inline__  __device__ double __log2u_frac( double x, double exp )
{
    bool pred = x > _SQRT_2;
    x *= pred ? 0.5 : _1_OVER_SQ2;
    double z = ( x - 1. ) * __rcp( x + 1. );
    double y = z * z * 33.9705627484771406 ;
    double s = _LOG2U_4;
    s = __fma_rn( s, y, _LOG2U_3 );
    s = __fma_rn( s, y, _LOG2U_2 );
    s = __fma_rn( s, y, _LOG2U_1 );
    s = __fma_rn( s, y, _LOG2U_0 );
    return __fma_rn( z, s, ( pred ? 1.0 : 0.5 ) + exp );
}

__inline__  __device__ int __log2u_intg( uint x )
{
    return 31 - __clz( x );
}

__inline__ __device__ double __log2u( uint x )
{
    int    I = __log2u_intg( x );
    return __log2u_frac( ( double )x * __2_to_n( -I ), I - 32 );
}

template<uint X> __inline__ __device__ uint __log2i()
{
    return 1U + __log2i < X / 2U > ();
}

template<> __inline__ __device__ uint __log2i<1U>()
{
    return 0U;
}

__inline__ __device__ uint __mantissa( float u, float v, float w )
{
    uint i = __float_as_uint( u ) & 0X7FF000U;
    uint j = __float_as_uint( v ) & 0X7FF000U;
    uint k = __float_as_uint( w ) & 0X7FF000U;
    return interleave3( i >> 12, j >> 12, k >> 12 );
}

#define _TEA_K0      0xA341316C
#define _TEA_K1      0xC8013EA4
#define _TEA_K2      0xAD90777D
#define _TEA_K3      0x7E95761E
#define _TEA_DT      0x9E3779B9

template<int N> __inline__ __host__ __device__ void __TEA_core( uint &v0, uint &v1, uint sum = 0 )
{
    sum += _TEA_DT;
    v0 += ( ( v1 << 4 ) + _TEA_K0 ) ^ ( v1 + sum ) ^ ( ( v1 >> 5 ) + _TEA_K1 );
    v1 += ( ( v0 << 4 ) + _TEA_K2 ) ^ ( v0 + sum ) ^ ( ( v0 >> 5 ) + _TEA_K3 );
    __TEA_core < N - 1 > ( v0, v1, sum );
}

template<> __inline__ __host__ __device__ void __TEA_core<0>( uint &v0, uint &v1, uint sum ) {}

template<int N> __inline__ __host__ __device__ uint premix_TEA( uint v0, uint v1 )
{
    __TEA_core<N>( v0, v1 );
    return v0 ^ v1;
}

template<int N> __inline__ __device__ double gaussian_TEA( bool pred, int u, int v )
{
    uint v0 =  pred ? u : v;
    uint v1 = !pred ? u : v;
    __TEA_core<N>( v0, v1 );
    double f = __cospi( ( v0 & 0X7FFFFFFF ) * _2_TO_MINUS_31 ) * ( ( v0 & 0X80000000 ) ? 1.0 : -1.0 );
    double r = __sqrtd( -2.0 * _LN_2 * __log2u( max( v1, 1 ) ) );
    return bound( r * f, -4.0, 4.0 );
}

template<int N> __inline__ __device__ float gaussian_TEA_fast( bool pred, int u, int v )
{
    uint v0 =  pred ? u : v;
    uint v1 = !pred ? u : v;
    __TEA_core<N>( v0, v1 );
    float f = sinpif( int( v0 ) * float(_2_TO_MINUS_31) );
    float r = sqrtf( -2.0f * float(_LN_2) * log2f( v1 * float(_2_TO_MINUS_32) ) );
    return bound( r * f, -4.0f, 4.0f );
}

template<int N> __inline__ __host__ __device__ double unsigned_TEA( bool pred, int u, int v, int seed0, int seed1 )
{
    pred_swap( pred, u, v );
    uint v0 = u ^ seed0;
    uint v1 = v ^ seed1;
    __TEA_core<N>( v0, v1 );
    return _SQRT_3 * ( ( v0 ^ v1 ) * _2_TO_MINUS_31 - 1.0 );
}

template<int N> __inline__ __host__ __device__ double uniform_TEA( uint v0, uint v1 )
{
    __TEA_core<N>( v0, v1 );
    return _SQRT_3 * ( ( v0 ^ v1 ) * _2_TO_MINUS_31 - 1.0 );
}

template<int N> __inline__ __host__ __device__ float uniform_TEA_fast( uint v0, uint v1 )
{
    __TEA_core<N>( v0, v1 );
    return ( v0 ^ v1 ) * float(_SQRT_3*_2_TO_MINUS_31) - float(_SQRT_3);
}

//__inline__  __device__  __host__ double LM(int u, int v, uint seed)
//{
//  unsigned long long m = ((unsigned long long) comp_n_swap(u, v) << 32) ^ seed;
//  m ^= 18446744073709551557ULL;
//  m  = m * 3935559000370003845ULL + 2691343689449507681ULL;
//  m ^= m >> 21; m ^= m << 37; m ^= m >> 4;
//  m *= 2685821657736338717ULL;
//  return _SQRT_3 * ( (m >> 32) * 4.656612873077392578125e-10 - 1.0 );
//}

__inline__ __device__ double fetch_double( texture<int2, cudaTextureType1D> tex, int i )
{
    int2 v = tex1Dfetch( tex, i );
    return __hiloint2double( v.y, v.x );
}

template<>
__inline__ __device__ double tex1Dfetch<double>( cudaTextureObject_t tex, int i )
{
    int2 v = tex1Dfetch<int2>( tex, i );
    return __hiloint2double( v.y, v.x );
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<double>( void )
{
    int e = ( int )sizeof( int ) * 8;

    return cudaCreateChannelDesc( e, e, 0, 0, cudaChannelFormatKindSigned );
}

__inline__ __device__ int __laneid()
{
    int laneid;
    asm volatile( "mov.s32 %0, %%laneid;" : "=r"( laneid ) );
    return laneid;
}

__inline__ __device__ uint __lanemask_lt()
{
    uint lanemask;
    asm volatile( "mov.u32 %0, %%lanemask_lt;" : "=r"( lanemask ) );
    return lanemask;
}

__inline__ __device__ int __warpid_local()
{
    return threadIdx.x / WARPSZ ;
}

__inline__ __device__ int __warpid_global()
{
    return ( blockIdx.x * blockDim.x + threadIdx.x ) / WARPSZ ;
}

__inline__ __device__ int __warp_num_local()
{
    return blockDim.x / WARPSZ ;
}

__inline__ __device__ int __warp_num_global()
{
    return ( gridDim.x * blockDim.x ) / WARPSZ ;
}

__inline__ __device__ bool is_first_active( uint ballot, uint laneid )
{
    return __popc( ballot << ( warpSize - laneid ) ) == 0 ;
}

template<typename T, typename S> __inline__ __device__ T atomic_add( T * addr, S const val ) {
	return atomicAdd( addr, T(val) );
}

#if __CUDA_ARCH__ < 600
template<typename S> __inline__ __device__ double atomic_add( double* address, S const val_ )
{
    double val = val_;
	unsigned long long int* address_as_ull = ( unsigned long long int* )address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed, __double_as_longlong( val + __longlong_as_double( assumed ) ) );
    } while( assumed != old );
    return old;
}
#endif

__inline__ __device__ void atomicMax( double* address, double val )
{
    unsigned long long int* address_as_ull = ( unsigned long long int* )address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed, __double_as_longlong( max( val, __longlong_as_double( assumed ) ) ) );
    } while( assumed != old );
}

__inline__ __device__ void atomicMax( float* address, float val )
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS( address_as_int, assumed, __float_as_int( max( val, __int_as_float( assumed ) ) ) );
    } while( assumed != old );
}

__inline__ __device__ void atomicMin( double* address, double val )
{
    unsigned long long int* address_as_ull = ( unsigned long long int* )address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed, __double_as_longlong( min( val, __longlong_as_double( assumed ) ) ) );
    } while( assumed != old );
}

__device__ __inline__ int __shfl_xor_( int var, int laneMask, int width = warpSize )
{
    return __shfl_xor( var, laneMask, width );
}
__device__ __inline__ float __shfl_xor_( float var, int laneMask, int width = warpSize )
{
    return __shfl_xor( var, laneMask, width );
}

__inline__ __device__ double __shfl_xor_( double var, int laneMask, int width = warpSize )
{
    int hi, lo;
    asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"( lo ), "=r"( hi ) : "d"( var ) );
    hi = __shfl_xor( hi, laneMask, width );
    lo = __shfl_xor( lo, laneMask, width );
    return __hiloint2double( hi, lo );
}

template<class TYPE> __device__ TYPE __warp_sum( TYPE value )
{
#pragma unroll
    for( int p = ( WARPSZ >> 1 ); p >= 1 ; p >>= 1 )
        value += __shfl_xor_( value, p );
    return value;
}

template<class TYPE> __device__ TYPE __warp_max( TYPE value )
{
#pragma unroll
    for( int p = ( WARPSZ >> 1 ); p >= 1 ; p >>= 1 )
        value = max( value, __shfl_xor_( value, p ) );
    return value;
}

template<class TYPE> __device__ TYPE __warp_min( TYPE value )
{
#pragma unroll
    for( int p = ( WARPSZ >> 1 ); p >= 1 ; p >>= 1 )
        value = min( value, __shfl_xor_( value, p ) );
    return value;
}

template <class TYPE>
__global__ void gpu_reduce_sum_dev( TYPE      *in,
                                    TYPE      *out,
                                    const uint len )
{
    TYPE val = 0;
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x ) val += in[i];
    val = __warp_sum( val );
    if( __laneid() == 0 ) atomic_add( out, val );
}

template <class TYPE>
__global__ void gpu_reduce_sum_host( TYPE *in, TYPE *out, const uint len )
{
    if( blockIdx.x > 0 ) return ;  // only one block is allowed
    __shared__ TYPE globalVal;
    if( threadIdx.x == 0 ) globalVal = 0;
    __syncthreads();

    TYPE val = 0 ;
    for( int p = threadIdx.x; p < len ; p += blockDim.x ) val += in[p];
    val = __warp_sum( val );
    if( __laneid() == 0 ) atomic_add( &globalVal, val );
    __syncthreads();

    if( threadIdx.x == 0 ) *out = globalVal;
}

template <class TYPE>
__global__ void gpu_reduce_max_dev( TYPE      *in,
                                    TYPE      *out,
                                    const TYPE mini,
                                    const uint len )
{
    TYPE val = mini;
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x ) val = max( val, in[i] );
    val = __warp_max( val );
    if( __laneid() == 0 ) atomicMax( out, val );
}

template <class TYPE>
__global__ void gpu_reduce_max_host( TYPE *in, TYPE *out, const TYPE mini, const uint len )
{
    if( blockIdx.x > 0 ) return ;  // only one block is allowed
    __shared__ TYPE global_val;
    if( threadIdx.x == 0 ) global_val = mini;
    __syncthreads();

    TYPE val = mini ;
    for( int p = threadIdx.x; p < len ; p += blockDim.x ) val = max( val, in[p] );
    val = __warp_max( val );
    if( __laneid() == 0 ) atomicMax( &global_val, val );
    __syncthreads();

    if( threadIdx.x == 0 ) *out = global_val;
}

__inline__ __device__ int __warp_prefix_incl( int value )
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .s32  t;\n\t"
        "   shfl.up.b32 t|p, %0, 0x1, 0x0;\n\t"
        "@p add.s32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x2, 0x0;\n\t"
        "@p add.s32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x4, 0x0;\n\t"
        "@p add.s32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x8, 0x0;\n\t"
        "@p add.s32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x10, 0x0;\n\t"
        "@p add.s32 %0, t, %0;\n\t"
        "}"
        : "+r"( value ) );
    return value;
}

__inline__ __device__ uint __warp_prefix_incl( uint value )
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u32  t;\n\t"
        "   shfl.up.b32 t|p, %0, 0x1, 0x0;\n\t"
        "@p add.u32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x2, 0x0;\n\t"
        "@p add.u32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x4, 0x0;\n\t"
        "@p add.u32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x8, 0x0;\n\t"
        "@p add.u32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x10, 0x0;\n\t"
        "@p add.u32 %0, t, %0;\n\t"
        "}"
        : "+r"( value ) );
    return value;
}

__inline__ __device__ float __warp_prefix_incl( float value )
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .f32  t;\n\t"
        "   shfl.up.b32 t|p, %0, 0x1, 0x0;\n\t"
        "@p add.f32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x2, 0x0;\n\t"
        "@p add.f32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x4, 0x0;\n\t"
        "@p add.f32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x8, 0x0;\n\t"
        "@p add.f32 %0, t, %0;\n\t"
        "   shfl.up.b32 t|p, %0, 0x10, 0x0;\n\t"
        "@p add.f32 %0, t, %0;\n\t"
        "}"
        : "+f"( value ) );
    return value;
}

template<class TYPE> __device__ TYPE __warp_prefix_excl( TYPE value )
{
    return __warp_prefix_incl( value ) - value;
}

template<class KEYTYPE, class VALTYPE> __device__ KEYTYPE __warp_bitonic( KEYTYPE key, VALTYPE val )
{
    KEYTYPE other;
    int lane_id = __laneid();

    other = __shfl_xor_( key, 0X1 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X1 ) ) ? other : key;

    other = __shfl_xor_( key, 0X3 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X2 ) ) ? other : key;
    other = __shfl_xor_( key, 0X1 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X1 ) ) ? other : key;

    other = __shfl_xor_( key, 0X7 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X4 ) ) ? other : key;
    other = __shfl_xor_( key, 0X3 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X2 ) ) ? other : key;
    other = __shfl_xor_( key, 0X1 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X1 ) ) ? other : key;

    other = __shfl_xor_( key, 0XF );
    key = ( ( other > key ) == ( bool )( lane_id & 0X8 ) ) ? other : key;
    other = __shfl_xor_( key, 0X7 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X4 ) ) ? other : key;
    other = __shfl_xor_( key, 0X3 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X2 ) ) ? other : key;
    other = __shfl_xor_( key, 0X1 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X1 ) ) ? other : key;

    other = __shfl_xor_( key, 0X1F );
    key = ( ( other > key ) == ( bool )( lane_id & 0X10 ) ) ? other : key;
    other = __shfl_xor_( key, 0XF );
    key = ( ( other > key ) == ( bool )( lane_id & 0X8 ) ) ? other : key;
    other = __shfl_xor_( key, 0X7 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X4 ) ) ? other : key;
    other = __shfl_xor_( key, 0X3 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X2 ) ) ? other : key;
    other = __shfl_xor_( key, 0X1 );
    key = ( ( other > key ) == ( bool )( lane_id & 0X1 ) ) ? other : key;
}

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDS_PTR   "l"
#else
#define __LDS_PTR   "r"
#endif

__device__ __inline__ char __lds( const char *ptr )
{
    unsigned int ret;
    asm volatile( "ld.global.cs.s8 %0, [%1];"  : "=r"( ret ) : __LDS_PTR( ptr ) );
    return ( char )ret;
}
__device__ __inline__ short __lds( const short *ptr )
{
    unsigned short ret;
    asm volatile( "ld.global.cs.s16 %0, [%1];"  : "=h"( ret ) : __LDS_PTR( ptr ) );
    return ( short )ret;
}
__device__ __inline__ int __lds( const int *ptr )
{
    unsigned int ret;
    asm volatile( "ld.global.cs.s32 %0, [%1];"  : "=r"( ret ) : __LDS_PTR( ptr ) );
    return ( int )ret;
}
__device__ __inline__ long long __lds( const long long *ptr )
{
    unsigned long long ret;
    asm volatile( "ld.global.cs.s64 %0, [%1];"  : "=l"( ret ) : __LDS_PTR( ptr ) );
    return ( long long )ret;
}
__device__ __inline__ unsigned char __lds( const unsigned char *ptr )
{
    unsigned int ret;
    asm volatile( "ld.global.cs.s8 %0, [%1];"  : "=r"( ret ) : __LDS_PTR( ptr ) );
    return ( unsigned char )ret;
}
__device__ __inline__ unsigned short __lds( const unsigned short *ptr )
{
    unsigned short ret;
    asm volatile( "ld.global.cs.s16 %0, [%1];"  : "=h"( ret ) : __LDS_PTR( ptr ) );
    return ret;
}
__device__ __inline__ unsigned int __lds( const unsigned int *ptr )
{
    unsigned int ret;
    asm volatile( "ld.global.cs.s32 %0, [%1];"  : "=r"( ret ) : __LDS_PTR( ptr ) );
    return ret;
}
__device__ __inline__ unsigned long long __lds( const unsigned long long *ptr )
{
    unsigned long long ret;
    asm volatile( "ld.global.cs.s64 %0, [%1];"  : "=l"( ret ) : __LDS_PTR( ptr ) );
    return ret;
}
__device__ __inline__ float __lds( const float *ptr )
{
    float ret;
    asm volatile( "ld.global.cs.f32 %0, [%1];"  : "=f"( ret ) : __LDS_PTR( ptr ) );
    return ret;
}
__device__ __inline__ double __lds( const double *ptr )
{
    double ret;
    asm volatile( "ld.global.cs.f64 %0, [%1];"  : "=d"( ret ) : __LDS_PTR( ptr ) );
    return ret;
}

#if (__CUDA_ARCH__<350)
template<typename TYPE> __device__ __inline__ TYPE __ldg( const TYPE *ptr )
{
    return *ptr;
}
#endif

#endif

#endif
