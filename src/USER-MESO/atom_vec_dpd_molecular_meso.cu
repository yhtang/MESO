#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"

#include "meso.h"
#include "atom_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_dpd_molecular_meso.h"

using namespace LAMMPS_NS;

void AtomVecDPDMolecular::grow_device( int nmax_new )
{
    MesoAtomVec::grow_device( nmax_new );

    meso_atom->dev_mole = dev_mole.grow( nmax_new );

    grow_exclusion();
}

// to capture the change of maxspecial
void AtomVecDPDMolecular::grow_exclusion()
{
    meso_device->sync_device();

    excl_table_padding        = ceiling( meso_atom->maxspecial, 8 );
    meso_atom->dev_nexcl      = dev_nexcl     .grow( meso_atom->nmax );
    meso_atom->dev_nexcl_full = dev_nexcl_full.grow( meso_atom->nmax * 3 );
    meso_atom->dev_excl_table = dev_excl_table.grow( meso_atom->nmax * excl_table_padding );

    dev_special_pinned.unmap_host( dev_special_pinned.ptr_h() );
    dev_special_pinned.map_host( meso_atom->nmax * meso_atom->maxspecial, &( meso_atom->special[0][0] ) );
}

void AtomVecDPDMolecular::pin_host_array()
{
    MesoAtomVec::pin_host_array();

    dev_mole_pinned   .map_host( meso_atom->nmax, meso_atom->molecule );
}

void AtomVecDPDMolecular::unpin_host_array()
{
    MesoAtomVec::unpin_host_array();

    dev_mole_pinned   .unmap_host( meso_atom->molecule );
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_nexcl(
    int4* __restrict dev,
    int*  __restrict host,
    int * __restrict perm_table,
    const int p_beg,
    const int nall )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if( tid < nall ) {
        int i = tid + p_beg;
        if( DIR == 0 ) {
            int4 n;
            int p = i;
            if( perm_table ) p = perm_table[p];
            n.x = host[ p * 3 + 0 ];
            n.y = host[ p * 3 + 1 ];
            n.z = host[ p * 3 + 2 ];
            dev[i] = n;
        } else {
            int4 n = dev[i];
            host[ i * 3 + 0 ] = n.x;
            host[ i * 3 + 1 ] = n.y;
            host[ i * 3 + 2 ] = n.z;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_excl(
    int4* __restrict n_excl,
    int*  __restrict host_excl_table,
    int*  __restrict device_excl_table,
    int*  __restrict perm_table,
    const int   host_padding,
    const r64   host_padding_inv,
    const int   device_padding,
    const int   p_beg,
    const int   nall
)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if( gid < nall * host_padding ) {
        if( DIR == 0 ) {
            int i = gid * host_padding_inv + p_beg;
            int p = gid - i * host_padding;
            if( p < n_excl[i].z ) {
                if( perm_table )
                    device_excl_table[ i * device_padding + p ] = host_excl_table[ perm_table[i] * host_padding + p ];
                else
                    device_excl_table[ i * device_padding + p ] = host_excl_table[ i * host_padding + p ];
            }
        } else {
            int i = gid * host_padding_inv + p_beg;
            int p = gid - i * host_padding;
            if( p < n_excl[i].z )
                host_excl_table[ i * host_padding + p ] = device_excl_table[ i * device_padding + p ];
        }
    }
}

CUDAEvent AtomVecDPDMolecular::transfer_exclusion( TransferDirection direction, int* permute_from, int p_beg, int n_atom, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if( action & ACTION_COPY ) dev_nexcl_full    .upload( &meso_atom->nspecial[p_beg][0], n_atom * 3, stream, p_beg * 3 );
        if( action & ACTION_COPY ) dev_special_pinned.upload( n_atom * meso_atom->maxspecial, stream, p_beg * meso_atom->maxspecial );
        int threads_per_block1 = meso_device->query_block_size( gpu_copy_nexcl<0> );
        int threads_per_block2 = meso_device->query_block_size( gpu_copy_excl<0> );
        if( action & ACTION_PERM )
            gpu_copy_nexcl<0> <<< n_block( n_atom, threads_per_block1 ), threads_per_block1, 0, stream>>>(
                dev_nexcl,
                dev_nexcl_full,
                permute_from,
                p_beg,
                n_atom );
        if( action & ACTION_PERM )
            gpu_copy_excl<0> <<< n_block( n_atom * meso_atom->maxspecial, threads_per_block2 ), threads_per_block2, 0, stream >>> (
                dev_nexcl,
                dev_special_pinned.buf_d(),
                dev_excl_table,
                permute_from,
                meso_atom->maxspecial,
                1.0 / meso_atom->maxspecial,
                excl_table_padding,
                p_beg,
                n_atom );
        e = meso_device->event( "AtomVecDPDMolecular::transfer_exclusion::C2G" );
        e.record( stream );
    } else {
        int threads_per_block1 = meso_device->query_block_size( gpu_copy_nexcl<1> );
        int threads_per_block2 = meso_device->query_block_size( gpu_copy_excl<1> );
        gpu_copy_nexcl<1> <<< n_block( n_atom, threads_per_block1 ), threads_per_block1, 0, stream>>>(
            dev_nexcl,
            dev_nexcl_full,
            NULL,
            p_beg,
            n_atom );
        gpu_copy_excl<1> <<< n_block( n_atom * meso_atom->maxspecial, threads_per_block2 ), threads_per_block2, 0, stream >>> (
            dev_nexcl,
            dev_special_pinned.buf_d(),
            dev_excl_table,
            NULL,
            meso_atom->maxspecial,
            1.0 / meso_atom->maxspecial,
            excl_table_padding,
            p_beg,
            n_atom );
        dev_nexcl_full.download( &meso_atom->nspecial[p_beg][0], n_atom * 3, stream, p_beg * 3 );
        dev_special_pinned.download( n_atom * meso_atom->maxspecial, stream, p_beg * meso_atom->maxspecial );
        e = meso_device->event( "AtomVecDPDMolecular::transfer_exclusion::G2C" );
        e.record( stream );
    }
    return e;
}

void AtomVecDPDMolecular::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    MesoAtomVec::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::MOLE ) {
        events.push_back(
            transfer_scalar(
                dev_mole_pinned, dev_mole, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::EXCL ) {
        events.push_back(
            transfer_exclusion( direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
}


