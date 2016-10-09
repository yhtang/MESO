#ifndef LMP_MESO_ATOM_VEC
#define LMP_MESO_ATOM_VEC

#include "atom_vec.h"
#include "atom_meso.h"
#include "meso.h"

namespace LAMMPS_NS
{

template<typename TYPE> __global__ void gpu_permute_copy(
    TYPE* __restrict out,
    TYPE* __restrict in,
    int * __restrict perm_table,
    const int p_beg,
    const int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i < n ) {
        if( perm_table ) out[p_beg + i] = in[ perm_table[p_beg + i] ];
        else out[p_beg + i] = in[p_beg + i];
    }
}

template<typename TYPE> __global__ void gpu_permute_copy_vector(
    TYPE** __restrict out,
    TYPE* __restrict in,
    int * __restrict perm_table,
    const int p_beg,
    const int nmax,
    const int n,
    const int d )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i < n ) {
		if( perm_table ) {
	    	for(int j = 0; j < d ; j++) out[ j ][ p_beg + i ] = in[ j * nmax + perm_table[p_beg + i] ];
		} else {
			for(int j = 0; j < d ; j++) out[ j ][ p_beg + i ] = in[ j * nmax + p_beg + i ];
		}
    }
}

template<int NLANE, typename TYPE, int CACHESIZE>
__global__ void gpu_deinterleave(
    TYPE * __restrict array_raw, /* layout: XYZ XYZ XYZ XYZ... */
    TYPE** __restrict lanes,     /* layout: XXXX...            */
    int * __restrict perm_table,
    const int  p_beg,
    const int  n_atom )
{
    __shared__ TYPE buffer[CACHESIZE];

    int g = blockIdx.x * blockDim.x * NLANE;
#pragma unroll
    for( int i = 0, p = threadIdx.x ; i < NLANE ; i++, p += blockDim.x )
        if( g + p < n_atom * NLANE ) buffer[ p ] = array_raw[ p_beg * NLANE + ( g + p ) ];
    __syncthreads();

    int p_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if( p_atom < n_atom ) {
        if( perm_table ) p_atom = perm_table[ p_beg + p_atom ];
#pragma unroll
        for( int i = 0 ; i < NLANE ; i++ )
            lanes[i][ p_beg + p_atom ] = buffer[ threadIdx.x * NLANE + i ];
    }
}

template<int NLANE, typename TYPE, int CACHESIZE>
__global__ void gpu_interleave(
    TYPE * __restrict array_raw, /* layout: XYZ XYZ XYZ XYZ... */
    TYPE** __restrict lanes,     /* layout: XXXX...            */
    const int p_beg,
    const int n_atom )
{
    __shared__ TYPE buffer[CACHESIZE];

    int p_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if( p_atom < n_atom ) {
        for( int i = 0 ; i < NLANE ; i++ )
            buffer[ threadIdx.x * NLANE + i ] = lanes[i][ p_beg + p_atom ];
    }
    __syncthreads();

    int g = blockIdx.x * blockDim.x * NLANE;
    for( int i = 0, p = threadIdx.x; i < NLANE ; i++, p += blockDim.x )
        if( g + p < n_atom * NLANE ) array_raw[ p_beg * NLANE + ( g + p ) ] = buffer[ p ];
}

template<typename TYPE>
__global__ void gpu_unpack_by_type( int  * __restrict type,
                                    TYPE * __restrict type_attr,
                                    TYPE * __restrict target,
                                    const int  n_target_type,
                                    const int  p_beg,
                                    const int  n_atom )
{
    extern __shared__ r64 mass_type_shared[];
    for( int i = threadIdx.x ; i < n_target_type; i += blockDim.x ) mass_type_shared[i] = type_attr[i];
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x ;
    if( gid < n_atom ) target[gid + p_beg] = mass_type_shared[ type[gid + p_beg] ];
}

class MesoAtomVec: protected MesoPointers
{
public:
    MesoAtomVec( class LAMMPS * );
    virtual ~MesoAtomVec();

    AtomAttribute::Descriptor pre_sort, post_sort, pre_border, post_border, pre_comm, post_comm, pre_exchange,  pre_output;
    int                       excl_table_padding;

    virtual void data_atom_target( int, double*, int, char** )
    {
        printf( "[CDEV] null function @ %s: %d\n", FLERR );
    }
    virtual void grow_device( int );

    virtual void pin_host_array();
    virtual void unpin_host_array();
    virtual void dp2sp_merged( int seed, int p_beg, int p_end, bool offset = false );
    virtual int  resolve_work_range( AtomAttribute::Descriptor per_atom_prop, int& n_beg, int& n_end );
    virtual void force_clear( AtomAttribute::Descriptor range, int vflag );
    virtual std::vector<CUDAEvent> transfer( AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int* permute_from = NULL, int* permute_to = NULL, int state = ACTION_COPY | ACTION_PERM, bool trainable = false );

protected:
    const static int    growth_inc;
    const static double growth_mul;
    int alloced;

    // permutation table for atom reordering
    DeviceScalar<u64> dev_perm_key;
    DeviceScalar<int> dev_permute_from;
    DeviceScalar<int> dev_permute_to;

    // per-atom properties
    DeviceScalar<int>    dev_tag;
    DeviceScalar<int>    dev_type;
    DeviceScalar<int>    dev_mask;
    DeviceScalar<r64>    dev_mass;
    DeviceVector<r64>    dev_coord;
    DeviceVector<r64>    dev_force;
    DeviceVector<r64>    dev_veloc;
    DeviceVector<r64>    dev_virial;
    DeviceScalar<tagint> dev_image;
    DeviceScalar<r64>    dev_e_pair;
    DeviceVector<r32>    dev_r_coord;
    DeviceVector<r32>    dev_r_veloc;
    DeviceScalar<float4> dev_coord_merged;
    DeviceScalar<float4> dev_veloc_merged;
    HostScalar<int>      hst_borderness;

    // pinned host memory, e.g. atom->x & pre-allocated device buffer
    Pinned<r64> dev_coord_pinned;
    Pinned<r64> dev_veloc_pinned;
    Pinned<r64> dev_force_pinned;
    Pinned<int> dev_image_pinned;
    Pinned<int> dev_tag_pinned;
    Pinned<int> dev_type_pinned;
    Pinned<int> dev_mask_pinned;
    Pinned<r64> dev_masstype_pinned;

    // *****anything between are dummy***
    DevicePitched<int>  devNImprop;
    DevicePitched<int>  devImprops, devImpropType;
    // *****anything between are dummy***

    virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed );

    // templated implementation of the transfer
    template<typename TYPE>
    CUDAEvent transfer_scalar( Pinned<TYPE> &pinned, DeviceScalar<TYPE> &array, TransferDirection direction,
                               int* permute_from, int p_beg, int n_atom, CUDAStream stream, int state, bool streamed = true )
    {
        CUDAEvent e;
        if( direction == CUDACOPY_C2G ) {
            int threads_per_block = meso_device->query_block_size( gpu_permute_copy<TYPE> );
            if( state & ACTION_COPY ) pinned.upload( n_atom, stream, p_beg );
            if( state & ACTION_PERM ) gpu_permute_copy <<< n_block( n_atom, threads_per_block ), threads_per_block, 0, stream>>>(
                    array.ptr(),
                    pinned.buf_d(),
                    permute_from,
                    p_beg,
                    n_atom );
            e = meso_device->event( "MesoAtomVec::transfer_scalar::C2G::" + array.tag() );
            if( streamed ) e.record( stream );
        } else {
            array.download( pinned.ptr_h() + p_beg, n_atom, stream, p_beg );
            e = meso_device->event( "MesoAtomVec::transfer_scalar::G2C::" + array.tag() );
            if( streamed ) e.record( stream );
        }
        return e;
    }
    template<int NLANE, typename TYPE, int BLOCKSIZE>
    CUDAEvent transfer_vector( Pinned<TYPE> &pinned, DeviceVector<TYPE> &array, TransferDirection direction,
                               int* permute_to, int p_beg, int n_atom, CUDAStream stream, int action, bool streamed = true )
    {
        CUDAEvent e;
        if( direction == CUDACOPY_C2G ) {
            if( action & ACTION_COPY ) pinned.upload( NLANE * n_atom, stream, NLANE * p_beg );
            if( action & ACTION_PERM ) gpu_deinterleave<NLANE, TYPE, BLOCKSIZE*NLANE>
                <<< n_block( n_atom, BLOCKSIZE ), BLOCKSIZE, 0, stream>>>(
                    pinned.buf_d(),
                    array.ptrs(),
                    permute_to,
                    p_beg,
                    n_atom );
            e = meso_device->event( "MesoAtomVec::transfer_vector::C2G::" + array.tag() );
            if( streamed ) e.record( stream );
        } else {
            gpu_interleave<NLANE, TYPE, BLOCKSIZE*NLANE>
            <<< n_block( n_atom, BLOCKSIZE ), BLOCKSIZE, 0, stream>>>(
                pinned.buf_d(),
                array.ptrs(),
                p_beg,
                n_atom );
            pinned.download( NLANE * n_atom, stream, NLANE * p_beg );
            e = meso_device->event( "MesoAtomVec::transfer_vector::G2C::" + array.tag() );
            if( streamed ) e.record( stream );
        }
        return e;
    }
    // for device vectors with dynamical dimension, no interleaving performed
    template<typename TYPE>
    CUDAEvent transfer_vector( Pinned<TYPE> &pinned, DeviceVector<TYPE> &array, TransferDirection direction,
                               int* permute_to, int p_beg, int n_atom, CUDAStream stream, int action, bool streamed = true )
    {
        CUDAEvent e;
        if( direction == CUDACOPY_C2G ) {
        	int threads_per_block = meso_device->query_block_size( gpu_permute_copy_vector<TYPE> );
            if( action & ACTION_COPY ) {
            	for(int d = 0 ; d < array.d() ; d++)
            		cudaMemcpyAsync( pinned.buf_d() + d * meso_atom->nmax + p_beg, pinned.ptr_h() + d * meso_atom->nmax + p_beg, n_atom * sizeof(TYPE), cudaMemcpyDefault, stream );
            }
            if( action & ACTION_PERM ) gpu_permute_copy_vector<TYPE>
                <<< n_block( n_atom, threads_per_block ), threads_per_block, 0, stream>>>(
                    array.ptrs(),
					pinned.buf_d(),
                    permute_to,
                    p_beg,
                    meso_atom->nmax,
                    n_atom,
                    array.d() );
            e = meso_device->event( "MesoAtomVec::transfer_vector::C2G::" + array.tag() );
            if( streamed ) e.record( stream );
        } else {
        	for(int d = 0 ; d < array.d() ; d++) {
        		cudaMemcpyAsync( pinned.ptr_h() + d * meso_atom->nmax + p_beg, array[d] + p_beg, n_atom * sizeof( TYPE ), cudaMemcpyDefault, stream );
        	}
            e = meso_device->event( "MesoAtomVec::transfer_vector::G2C::" + array.tag() );
            if( streamed ) e.record( stream );
        }
        return e;
    }
    template<typename TYPE>
    CUDAEvent unpack_by_type( Pinned<TYPE> &type_attr, DeviceScalar<int> &type, DeviceScalar<TYPE> &target, int ntypes,
                              TransferDirection direction, int p_beg, int n_atom, CUDAStream stream, int action, bool streamed = true )
    {
        // permutable table not used, because type has already been permuted
        CUDAEvent e;
        if( direction == CUDACOPY_C2G ) {
            int threads_per_block = meso_device->query_block_size( gpu_unpack_by_type<TYPE> );
            if( action & ACTION_COPY ) type_attr.upload( ntypes, stream );
            if( action & ACTION_PERM ) gpu_unpack_by_type <<< n_block( n_atom, threads_per_block ), threads_per_block, ntypes*sizeof( TYPE ), stream>>>
                ( type.ptr(), type_attr.buf_d(), target.ptr(), ntypes, p_beg, n_atom );
            e = meso_device->event( "MesoAtomVec::unpack_by_type::C2G::" + target.tag() );
            if( streamed ) e.record( stream );
        } else {
            // nothing to be done
            e = meso_device->event( "MesoAtomVec::unpack_by_type::G2C::" + target.tag() );
            if( streamed ) e.record( stream );
        }
        return e;
    }
};

}

#endif
