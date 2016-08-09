#ifndef LMP_MESO_PREFIX
#define LMP_MESO_PREFIX

// specialized prefix sum kernel
// 1G bit prefix sum by 3 passes

#include "meso.h"


namespace LAMMPS_NS
{

// digest arbitrary array into at most 1024 sectors
template<class TYPE> __global__ void gpu_bit_prefix_down( uint *in, TYPE *out, TYPE *next, const int jobsize, const int njob, const int n )
{
    __shared__ int previous_sum;
    __shared__ int buffer[32];

    for( int j = blockIdx.x; j < njob; j += gridDim.x ) {
        if( threadIdx.x == 0 ) previous_sum = 0;
        __syncthreads();

        for( int i = threadIdx.x ; i < jobsize ; i += blockDim.x ) {
            // ld
            int p = i + j * jobsize;
            int v = ( p < n ) ? ( __popc( in[p] ) ) : 0;

            // intra-warp prefix
            int prefix_warp = __warp_prefix_excl( v );
            if( __laneid() == 31 ) buffer[ threadIdx.x / 32 ] = prefix_warp + v ;  // + x : inclusive result
            __syncthreads();

            // inter-warp prefix
            if( threadIdx.x < 32 ) buffer[ threadIdx.x ] = __warp_prefix_excl( buffer[threadIdx.x] );
            __syncthreads();

            // intra-warp shift & st
            int prefix_block = prefix_warp + buffer[ threadIdx.x / 32 ];
            if( p < n ) out[p] = prefix_block + previous_sum ;
            __syncthreads();
            if( threadIdx.x == blockDim.x - 1 ) previous_sum += prefix_block + v ;
        }
        __syncthreads();

        if( threadIdx.x == 0 ) next[ j ] = previous_sum;
        __syncthreads();
    }
}

// single CTA 1024 element prefix sum
template<class TYPE> __global__ void gpu_bit_prefix_block( TYPE *val, const int n )
{
    __shared__ int buffer[32];

    // ld & intra-warp prefix
    int v = ( threadIdx.x < n ) ? ( val[threadIdx.x] ) : 0;
    int prefix_warp = __warp_prefix_excl( v );
    if( __laneid() == 31 ) buffer[ threadIdx.x / 32 ] = prefix_warp + v ;  // + x : inclusive result
    __syncthreads();

    // inter-warp prefix
    if( threadIdx.x < 32 ) buffer[ threadIdx.x ] = __warp_prefix_excl( buffer[threadIdx.x] );
    __syncthreads();

    // intra-warp shift & st
    if( threadIdx.x < n ) val[threadIdx.x] = prefix_warp + buffer[ threadIdx.x / 32 ];
    __syncthreads();
}

// broadcast the single CTA result back to original array
template<class TYPE> __global__ void gpu_bit_prefix_up( TYPE *out, TYPE *next, const int work, const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x ; i < n ; i += gridDim.x * blockDim.x ) {
        out[i] += next[ i / work ];
    }
}

template<class TYPE> class MesoPrefixPlan : protected Pointers, protected MesoPointers
{
public:
    MesoPrefixPlan( LAMMPS *lmp ) : Pointers( lmp ), MesoPointers( lmp ),
        L1( NULL ),
        L2( NULL ),
        GridCfg1( make_int2( 0, 0 ) ),
        GridCfg2( make_int2( 0, 0 ) ),
        GridCfg3( make_int2( 0, 0 ) )
    {
        CreatePlan( 1024 * 1024 );
    }
    MesoPrefixPlan( const MesoPrefixPlan<TYPE> &another ) : // copy constructor
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        L1( NULL ),
        L2( NULL ),
        GridCfg1( make_int2( 0, 0 ) ),
        GridCfg2( make_int2( 0, 0 ) ),
        GridCfg3( make_int2( 0, 0 ) )
    {
        *this = another;
    }
    MesoPrefixPlan<TYPE>& operator = ( const MesoPrefixPlan<TYPE> &another ) // copy assignment operator
    {
        DestroyPlan();
        CreatePlan( another._n );
        return *this;
    }
    ~MesoPrefixPlan()
    {
        DestroyPlan();
    }
    int* prefix( uint *bits, int n, CUDAStream stream )
    {
        if( _n < n ) {
            stream.sync();
            DestroyPlan();
            CreatePlan( n * 1.2 );
        }
        GridConfig();
        int nInt = ceiling( n, 32 ) / 32;
        int job_size = ( nInt < GridCfg1.y * GridCfg2.y ) ? ( GridCfg1.y ) : ceiling( nInt / GridCfg2.y, GridCfg1.y ) ;
        int njob = n_block( nInt, job_size );
        gpu_bit_prefix_down <<< GridCfg1.x, GridCfg1.y, 0, stream >>>( bits, L1, L2, job_size, njob, nInt );
        gpu_bit_prefix_block <<<          1, GridCfg2.y, 0, stream >>>( L2, n_block( nInt, job_size ) );
        gpu_bit_prefix_up   <<< GridCfg3.x, GridCfg3.y, 0, stream >>>( L1, L2, job_size, nInt );
        return L1;
    }
public:
    int _n;
    TYPE *L1, *L2;
    int2 GridCfg1, GridCfg2, GridCfg3;

    void CreatePlan( int n ) // number of bits
    {
        if( !n ) return;
        _n = n;
        meso_device->realloc_device( "MesoPrefixPlan::L1", L1, ceiling( n, 32 ) / 32, false, false );
        meso_device->realloc_device( "MesoPrefixPlan::L2", L2, 1024, false, false );
    }
    void DestroyPlan()
    {
        _n = 0;
        if( L1 ) meso_device->free( L1 );
        if( L2 ) meso_device->free( L2 );
    }
    void GridConfig()
    {
        if( !GridCfg1.x ) {
            GridCfg1 = meso_device->occu_calc.right_peak( 0, gpu_bit_prefix_down <TYPE>, 0, cudaFuncCachePreferShared );
            GridCfg2 = meso_device->occu_calc.right_peak( 0, gpu_bit_prefix_block<TYPE>, 0, cudaFuncCachePreferShared );
            GridCfg3 = meso_device->occu_calc.right_peak( 0, gpu_bit_prefix_up   <TYPE>, 0, cudaFuncCachePreferShared );
        }
    }
};

}

#endif
