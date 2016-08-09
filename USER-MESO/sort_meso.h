#ifndef LMP_MESO_SORT
#define LMP_MESO_SORT

#include "meso.h"


namespace LAMMPS_NS
{

inline __host__ __device__ double LOG2( double x )
{
    return log( x ) / log( 2.0 );
}
template<typename T> inline __device__ void __swap( T &a, T &b )
{
    T tmp = a;
    a = b;
    b = tmp;
}

template<int RADIX, typename TYPE>
__global__ void gpu_radix_histogram(
    TYPE* __restrict key,
    int*  __restrict tprev,
    int*  __restrict wprev,
    const int len,
    const int bit,
    const int prev_padding
)
{
    __shared__ int *wLocal[RADIX];
    if( threadIdx.x < RADIX ) wLocal[ threadIdx.x ] = wprev + threadIdx.x * prev_padding ;
    __syncthreads();

    int laneID = __laneid();
    for( int i = threadIdx.x + blockIdx.x * blockDim.x ; i < len ; i += blockDim.x * gridDim.x ) {
        int radix = ( key[i] >> bit ) & ( RADIX - 1 );
#pragma unroll
        for( int r = 0 ; r < RADIX ; r++ ) {
            uint before = __ballot( radix == r );
            if( radix  == r ) tprev[i] = __popc( before << ( WARPSZ - laneID ) );
            if( laneID == 0 ) wLocal[ r ][ i / WARPSZ ] = __popc( before ) ;
        }
    }
}

template<typename TYPE> __global__ void gpu_radix_prefix_sum( TYPE *vals, const int n, const int padding )
{
    device_static_assert( WARPSZ == 32, WARP_SIZE_MUST_BE_32_FOR_RADIX_SORT );

    __shared__ TYPE previous_sum;
    __shared__ TYPE buffer[32];
    if( threadIdx.x == 0 ) previous_sum = 0;
    __syncthreads();

    int len = ( n + blockDim.x - 1 ) & ( ~( blockDim.x - 1 ) );
    TYPE *val = vals + blockIdx.x * padding;
    for( int i = threadIdx.x ; i < len ; i += blockDim.x ) {
        // ld
        TYPE in;
        if( i < n ) in = val[i];
        else in = 0;

        // intra-warp prefix
        TYPE sum_warp = __warp_prefix_excl( in );
        if( __laneid() == 31 ) buffer[ threadIdx.x / 32 ] = sum_warp + in ;  // + x : inclusive result
        __syncthreads();

        // inter-warp prefix
        if( threadIdx.x < 32 ) buffer[ threadIdx.x ] = __warp_prefix_excl( buffer[threadIdx.x] );
        __syncthreads();

        // intra-warp shift & st
        TYPE sum_block = sum_warp + buffer[ threadIdx.x / 32 ];
        if( i < n ) val[i] = sum_block + previous_sum ;
        __syncthreads();
        if( threadIdx.x == blockDim.x - 1 ) previous_sum += sum_block + in ;
    }
    __syncthreads();
    if( threadIdx.x == 0 ) val[padding - 1] = previous_sum;
}

template<int RADIX,
         typename KeyType,
         typename ValType>
void __global__ gpu_radix_permute(
    KeyType* __restrict key,
    ValType* __restrict val,
    KeyType* __restrict key_out,
    ValType* __restrict val_out,
    int* __restrict tprev,
    int* __restrict wprev,
    const int len,
    const int bit,
    const int prev_padding
)
{
    // get_num_of_total_ones
    __shared__ int totals[RADIX];
    if( threadIdx.x == 0 ) {
        totals[0] = 0;
        for( int p = 1 ; p < RADIX ; p++ ) totals[p] = totals[p - 1] + wprev[ p * prev_padding - 1 ];
    }
    __syncthreads();

    for( int i = threadIdx.x + blockIdx.x * blockDim.x ; i < len ; i += blockDim.x * gridDim.x ) {
        int radix = ( key[i] >> bit ) & ( RADIX - 1 );
        int new_i = totals[radix] + tprev[i] + wprev[ radix * prev_padding + ( i / WARPSZ ) ] ;
        key_out[ new_i ] = key[i];
        val_out[ new_i ] = val[i];
    }
}

template<int RADIX, typename TYPE>
__forceinline__ __device__ void radix_histogram_singleblock(
    TYPE* __restrict key,
    int*  __restrict tprev,
    int*  __restrict wprev,
    const int len,
    const int bit,
    const int prev_padding
)
{
    __shared__ int *wLocal[RADIX];
    if( threadIdx.x < RADIX ) wLocal[ threadIdx.x ] = wprev + threadIdx.x * prev_padding ;
    __syncthreads();

    int laneID = __laneid();
    for( int i = threadIdx.x; i < len ; i += blockDim.x ) {
        int radix = ( key[i] >> bit ) & ( RADIX - 1 );
#pragma unroll
        for( int r = 0 ; r < RADIX ; r++ ) {
            uint before = __ballot( radix == r );
            if( radix  == r ) tprev[i] = __popc( before << ( WARPSZ - laneID ) );
            if( laneID == 0 ) wLocal[ r ][ i / WARPSZ ] = __popc( before ) ;
        }
    }
}

template<int RADIX, typename TYPE>
__forceinline__ __device__ void radix_prefix_sum_singleblock( TYPE *vals, const int n, const int r, const int padding )
{
    device_static_assert( WARPSZ == 32, WARP_SIZE_MUST_BE_32_FOR_RADIX_SORT );
    __shared__ TYPE previous_sum;
    __shared__ TYPE buffer[WARPSZ];
    int len = ( n + blockDim.x - 1 ) & ( ~( blockDim.x - 1 ) );

//  for(int r = 0; r < RADIX ; r++)
//  {
    if( threadIdx.x == 0 ) previous_sum = 0;
    __syncthreads();

    TYPE *val = vals + r * padding;
    for( int i = threadIdx.x ; i < len ; i += blockDim.x ) {
        // ld
        TYPE in;
        if( i < n ) in = val[i];
        else in = 0;

        // intra-warp prefix
        TYPE sum_warp = __warp_prefix_excl( in );
        if( __laneid() == WARPSZ - 1 ) buffer[ threadIdx.x / WARPSZ ] = sum_warp + in ; // + x : inclusive result
        __syncthreads();

        // inter-warp prefix
        if( threadIdx.x < WARPSZ ) buffer[ threadIdx.x ] = __warp_prefix_excl( buffer[threadIdx.x] );
        __syncthreads();

        // intra-warp shift & st
        TYPE sum_block = sum_warp + buffer[ threadIdx.x / 32 ];
        if( i < n ) val[i] = sum_block + previous_sum ;
        __syncthreads();
        if( threadIdx.x == blockDim.x - 1 ) previous_sum += sum_block + in ;
    }

    __syncthreads();
    if( threadIdx.x == 0 ) val[padding - 1] = previous_sum;
//  }
}

template<int RADIX, typename KeyType, typename ValType>
__forceinline__ __device__ void radix_permute_singleblock(
    KeyType* __restrict key,
    ValType* __restrict val,
    KeyType* __restrict key_out,
    ValType* __restrict val_out,
    int* __restrict tprev,
    int* __restrict wprev,
    const int len,
    const int bit,
    const int prev_padding
)
{
    // get_num_of_total_ones
    __shared__ int totals[RADIX];
    if( threadIdx.x == 0 ) {
        totals[0] = 0;
        for( int p = 1 ; p < RADIX ; p++ ) totals[p] = totals[p - 1] + wprev[ p * prev_padding - 1 ];
    }
    __syncthreads();

    for( int i = threadIdx.x; i < len ; i += blockDim.x ) {
        int radix = ( key[i] >> bit ) & ( RADIX - 1 );
        int new_i = totals[radix] + tprev[i] + wprev[ radix * prev_padding + ( i / WARPSZ ) ] ;
        key_out[ new_i ] = key[i];
        val_out[ new_i ] = val[i];
    }
}

template<int RADIX, typename KeyType, typename ValType>
__global__ void gpu_sort_singleblock(
    KeyType* __restrict key,
    ValType* __restrict val,
    KeyType* __restrict key_out,
    ValType* __restrict val_out,
    int* __restrict thread_prev,
    int* __restrict warp_prev,
    const int warp_padding,
    const int n,
    const uint n_bit )
{
    KeyType *key_in = key, *key_ou = key_out;
    ValType *val_in = val, *val_ou = val_out;
    for( uint bit = 0; bit < n_bit; bit += __log2i<RADIX>() ) {
        radix_histogram_singleblock<RADIX, KeyType>( key_in, thread_prev, warp_prev, n, bit, warp_padding );
        __syncthreads();
        for( int r = 0 ; r < RADIX ; r++ ) {
            radix_prefix_sum_singleblock<RADIX, int>( warp_prev, ( n + WARPSZ - 1 ) / WARPSZ, r, warp_padding );
            __syncthreads();
        }
        radix_permute_singleblock<RADIX, KeyType, ValType>( key_in, val_in, key_ou, val_ou, thread_prev, warp_prev, n, bit, warp_padding );
        __syncthreads();
        __swap( key_in, key_ou );
        __swap( val_in, val_ou );
    }
    if( key != key_in ) {
        for( int i = threadIdx.x; i < n; i += blockDim.x ) {
            key[i] = key_in[i];
            val[i] = val_in[i];
        }
    }
}

template<typename KeyType, typename ValType>
__global__ void gpu_sort_binary_singleblock(
    KeyType* __restrict key,
    ValType* __restrict val,
    KeyType* __restrict key_out,
    ValType* __restrict val_out,
    int* __restrict thread_prev,
    int* __restrict warp_prev,
    const int warp_padding,
    const int n,
    const uint n_bit )
{
    device_static_assert( WARPSZ == 32, WARP_SIZE_MUST_BE_32_FOR_RADIX_SORT );

    KeyType *key_in = key, *key_ou = key_out;
    ValType *val_in = val, *val_ou = val_out;

    for( uint bit = 0; bit < n_bit; bit++ ) {
        // Build histogram
        int laneID = __laneid();
        for( int i = threadIdx.x; i < n ; i += blockDim.x ) {
            int radix = ( key_in[i] >> bit ) & 1;
            uint _1_before = __ballot(  radix );
            uint _0_before = __ballot( !radix );//~ _1_before;
            int shift = WARPSZ - laneID;
            thread_prev[i] = radix ? __popc( _1_before << shift ) : __popc( _0_before << shift );
            if( laneID == 0 ) {
                warp_prev[ i / WARPSZ ] = __popc( _0_before ) ;
                warp_prev[ warp_padding + i / WARPSZ ] = __popc( _1_before ) ;
            }
        }
//      __syncthreads(); // saved by the one below

        // prefix sum

        __shared__ int _0_previous_sum, _1_previous_sum, _0_total;
        __shared__ int buffer[2][WARPSZ];
        int N = ( n + WARPSZ - 1 ) / WARPSZ;
        int len = ( N + blockDim.x - 1 ) & ( ~( blockDim.x - 1 ) );

        if( threadIdx.x == 0 ) _0_previous_sum = _1_previous_sum = 0;
        __syncthreads();

        for( int i = threadIdx.x ; i < len ; i += blockDim.x ) {
        	// ld
            int _0_in, _1_in;
            if( i < N ) {
                _0_in = warp_prev[i];
                _1_in = warp_prev[warp_padding + i];
            } else {
                _0_in = 0;
                _1_in = 0;
            }

            // intra-warp prefix
            int _0_sum_warp = __warp_prefix_excl( _0_in );
            int _1_sum_warp = __warp_prefix_excl( _1_in );
            if( laneID == WARPSZ - 1 ) {
                buffer[0][ threadIdx.x / WARPSZ ] = _0_sum_warp + _0_in ; // + x : inclusive result
                buffer[1][ threadIdx.x / WARPSZ ] = _1_sum_warp + _1_in ; // + x : inclusive result
            }
            __syncthreads();

            // inter-warp prefix
            if( threadIdx.x < WARPSZ ) {
                buffer[0][ threadIdx.x ] = __warp_prefix_excl( buffer[0][threadIdx.x] );
                buffer[1][ threadIdx.x ] = __warp_prefix_excl( buffer[1][threadIdx.x] );
            }
            __syncthreads();

            // intra-warp shift & st
            int _0_sum_block = _0_sum_warp + buffer[0][ threadIdx.x / WARPSZ ];
            int _1_sum_block = _1_sum_warp + buffer[1][ threadIdx.x / WARPSZ ];
            if( i < N ) {
                warp_prev[i] = _0_sum_block + _0_previous_sum ;
                warp_prev[warp_padding + i] = _1_sum_block + _1_previous_sum ;
            }
            __syncthreads();
            if( threadIdx.x == blockDim.x - 1 ) {
                _0_previous_sum += _0_sum_block + _0_in ;
                _1_previous_sum += _1_sum_block + _1_in ;
                _0_total = _0_previous_sum;
            }
            __syncthreads();
        }

        // permute
        for( int i = threadIdx.x; i < n ; i += blockDim.x ) {
            int radix = ( key_in[i] >> bit ) & 1;
            int new_i = ( radix ? _0_total : 0 ) + thread_prev[i] + warp_prev[( radix ? warp_padding : 0 ) + ( i / WARPSZ ) ];
            key_ou[ new_i ] = key_in[i];
            val_ou[ new_i ] = val_in[i];
        }
        __syncthreads();

        __swap( key_in, key_ou );
        __swap( val_in, val_ou );
    }
    if( key != key_in ) {
        for( int i = threadIdx.x; i < n; i += blockDim.x ) {
            key[i] = key_in[i];
            val[i] = val_in[i];
        }
    }
}

template<int RADIX, class KeyType, class ValType> class SortPlan : protected MesoPointers
{
public:
    SortPlan( LAMMPS *lmp ) : MesoPointers( lmp ),
        thread_prev( lmp, "SortPlan::ThreadPrev" ),
        warp_prev( lmp, "SortPlan::WarpPrev" ),
        key_out( lmp, "SortPlan::KeyOut" ),
        val_out( lmp, "SortPlan::ValOut" )
    {
        GridCfg1 = meso_device->configure_kernel( gpu_radix_histogram<RADIX, KeyType>,       0, true, cudaFuncCachePreferShared );
        GridCfg2 = meso_device->configure_kernel( gpu_radix_prefix_sum<int>,                0, true, cudaFuncCachePreferShared );
        GridCfg3 = meso_device->configure_kernel( gpu_radix_permute<RADIX, KeyType, ValType>, 0, true, cudaFuncCachePreferShared );
        GridCfg4 = make_int2( 1, meso_device->query_block_size( gpu_sort_singleblock<RADIX, KeyType, ValType> ) );
        meso_device->configure_kernel( gpu_sort_singleblock<RADIX, KeyType, ValType>,  0, true, cudaFuncCachePreferShared );
        CreatePlan( 1024 );
    }
    ~SortPlan()
    {
        DestroyPlan();
    }
    void sort( DeviceScalar<KeyType> &key, DeviceScalar<ValType> &val, int n, int n_bit, CUDAStream stream )
    {
        if( _n < n || _n > n * 2 ) {
            stream.sync();
            DestroyPlan();
            CreatePlan( n * 1.2 );
        }
        KeyType *key_in = key.ptr();
        ValType *val_in = val.ptr();
        KeyType *key_ou = key_out.ptr();
        ValType *val_ou = val_out.ptr();
        for( int bit = 0; bit < n_bit; bit += LOG2( RADIX ) ) {
            gpu_radix_histogram<RADIX, KeyType>       <<< GridCfg1.x, GridCfg1.y, 0, stream>>>( key_in, thread_prev, warp_prev, n, bit, warp_padding );
            gpu_radix_prefix_sum                      <<<      RADIX, GridCfg2.y, 0, stream>>>( warp_prev.ptr(), ( n + WARPSZ - 1 ) / WARPSZ, warp_padding );
            gpu_radix_permute<RADIX, KeyType, ValType> <<< GridCfg3.x, GridCfg3.y, 0, stream>>>( key_in, val_in, key_ou, val_ou, thread_prev, warp_prev, n, bit, warp_padding );
            std::swap( key_in, key_ou );
            std::swap( val_in, val_ou );
        }
        if( key.ptr() != key_in ) {
            key.upload( key_in, n, stream );
            val.upload( val_in, n, stream );
        }
    }
    void sort_singleblock( DeviceScalar<KeyType> &key, DeviceScalar<ValType> &val, int n, int n_bit, CUDAStream stream )
    {
        if( _n < n || _n > n * 2 ) {
            stream.sync();
            DestroyPlan();
            CreatePlan( n * 1.2 );
        }
        gpu_sort_binary_singleblock<KeyType, ValType> <<< GridCfg4.x, GridCfg4.y, 0, stream>>>(
            key.ptr(), val.ptr(), key_out.ptr(), val_out.ptr(), thread_prev.ptr(), warp_prev.ptr(), warp_padding, n, n_bit );
    }
protected:
    int _n, warp_padding;
    DeviceScalar<int> thread_prev; // radix counting buffer
    DeviceScalar<int> warp_prev; // radix counting buffer
    DeviceScalar<KeyType> key_out;
    DeviceScalar<ValType> val_out; // ping-pong buffer
    int2 GridCfg1, GridCfg2, GridCfg3, GridCfg4;

    void CreatePlan( int n )
    {
        if( !n ) return;
        _n = n;
        warp_padding = ( n + WARPSZ - 1 ) / WARPSZ + 1;
        key_out.grow( n );
        val_out.grow( n );
        thread_prev.grow( n );
        warp_prev.grow( warp_padding * RADIX );
    }
    void DestroyPlan()
    {
        _n = warp_padding = 0;
        key_out.grow( 0 );
        val_out.grow( 0 );
        thread_prev.grow( 0 );
        warp_prev.grow( 0 );
    }
};

}

#endif
