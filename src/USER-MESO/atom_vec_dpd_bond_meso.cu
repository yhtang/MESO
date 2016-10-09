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
#include "atom_vec_dpd_bond_meso.h"

using namespace LAMMPS_NS;

#define DELTA 10000

AtomVecDPDBond::AtomVecDPDBond( LAMMPS *lmp ) :
    AtomVecBond( lmp ),
    AtomVecDPDMolecular( lmp ),
    dev_e_bond( lmp, "AtomVecDPDBond::dev_e_bond" ),
    dev_nbond( lmp, "AtomVecDPDBond::dev_nbond" ),
    dev_bond( lmp, "AtomVecDPDBond::dev_bond" ),
    dev_bond_mapped( lmp, "AtomVecDPDBond::dev_bond_mapped" ),
    dev_nbond_pinned( lmp, "AtomVecDPDBond::dev_nbond_pinned" ),
    dev_bond_atom_pinned( lmp, "AtomVecDPDBond::dev_bond_atom_pinned" ),
    dev_bond_type_pinned( lmp, "AtomVecDPDBond::dev_bond_type_pinned" )
{
    cudable = 1;
    comm_x_only = 0;

    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE |
                   AtomAttribute::EXCL   | AtomAttribute::BOND;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD    | AtomAttribute::VELOC;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD    | AtomAttribute::VELOC;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE |
                   AtomAttribute::EXCL   | AtomAttribute::BOND;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE |
                   AtomAttribute::MOLE   | AtomAttribute::EXCL     | AtomAttribute::BOND;
}

void AtomVecDPDBond::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecDPDBond::grow_reset()
{
    AtomVecBond::grow_reset();
    grow_exclusion();
}

void AtomVecDPDBond::grow_cpu( int n )
{
    AtomVecBond::grow( n );
}

void AtomVecDPDBond::grow_device( int nmax_new )
{
    AtomVecDPDMolecular::grow_device( nmax_new );

    meso_atom->dev_e_bond       = dev_e_bond      .grow( nmax_new, false, true );
    meso_atom->dev_nbond        = dev_nbond       .grow( nmax_new );
    meso_atom->dev_bond        = dev_bond       .grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_mapped = dev_bond_mapped.grow( nmax_new , atom->bond_per_atom );
}

void AtomVecDPDBond::pin_host_array()
{
    AtomVecDPDMolecular::pin_host_array();

    if( atom->bond_per_atom ) {
        if( atom->num_bond ) dev_nbond_pinned    .map_host( atom->nmax, atom->num_bond );
        if( atom->bond_atom ) dev_bond_atom_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_atom[0][0] ) );
        if( atom->bond_type ) dev_bond_type_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_type[0][0] ) );
    }
}

void AtomVecDPDBond::unpin_host_array()
{
    AtomVecDPDMolecular::unpin_host_array();

    dev_nbond_pinned    .unmap_host( atom->num_bond );
    dev_bond_atom_pinned.unmap_host( atom->bond_atom ? atom->bond_atom[0] : NULL );
    dev_bond_type_pinned.unmap_host( atom->bond_type ? atom->bond_type[0] : NULL );
}

void AtomVecDPDBond::data_atom_target( int i, double *coord, int imagetmp, char **values )
{
    tag[i] = atoi( values[0] );
    if( tag[i] <= 0 )
        error->one( FLERR, "Invalid atom ID in Atoms section of data file" );

    molecule[i] = atoi( values[1] );

    type[i] = atoi( values[2] );
    if( type[i] <= 0 || type[i] > atom->ntypes )
        error->one( FLERR, "Invalid atom type in Atoms section of data file" );

    x[i][0] = coord[0];
    x[i][1] = coord[1];
    x[i][2] = coord[2];

    image[i] = imagetmp;

    mask[i] = 1;
    v[i][0] = 0.0;
    v[i][1] = 0.0;
    v[i][2] = 0.0;
    num_bond[i] = 0;
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_bond(
    int*  __restrict host_n_bond,
    int*  __restrict host_bond_atom,
    int*  __restrict host_bond_type,
    int*  __restrict n_bond,
    int2* __restrict bonds,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_bond[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int2 bond;
                bond.x = host_bond_atom[ i * padding_host + p ];
                bond.y = host_bond_type[ i * padding_host + p ];
                bonds[ tid + p * padding_device ] = bond;
            }
            n_bond[ tid ] = n;
        } else {
            int n = n_bond[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int2 bond = bonds[ i + p * padding_device ];
                host_bond_atom[ i * padding_host + p ] = bond.x;
                host_bond_type[ i * padding_host + p ] = bond.y;
            }
            host_n_bond[ i ] = n;
        }
    }
}

CUDAEvent AtomVecDPDBond::transfer_bond( TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if( action & ACTION_COPY ) dev_nbond_pinned.upload( n_transfer, stream, p_beg );
        if( action & ACTION_COPY ) dev_bond_atom_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        if( action & ACTION_COPY ) dev_bond_type_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        int threads_per_block = meso_device->query_block_size( gpu_copy_bond<0> );
        if( action & ACTION_PERM )
            gpu_copy_bond<0> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
                dev_nbond_pinned.buf_d(),
                dev_bond_atom_pinned.buf_d(),
                dev_bond_type_pinned.buf_d(),
                dev_nbond,
                dev_bond,
                permute_from,
                atom->bond_per_atom,
                dev_bond.pitch_elem(),
                p_beg,
                n_transfer );
        e = meso_device->event( "AtomVecDPDBond::transfer_bond::C2G" );
        e.record( stream );
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_bond<1> );
        gpu_copy_bond<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_nbond_pinned.buf_d(),
            dev_bond_atom_pinned.buf_d(),
            dev_bond_type_pinned.buf_d(),
            dev_nbond,
            dev_bond,
            NULL,
            atom->bond_per_atom,
            dev_bond.pitch_elem(),
            p_beg,
            n_transfer );
        dev_nbond_pinned.download( n_transfer, stream, p_beg );
        dev_bond_atom_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        dev_bond_type_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        e = meso_device->event( "AtomVecDPDBond::transfer_bond::G2C" );
        e.record( stream );
    }
    return e;
}

void AtomVecDPDBond::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    AtomVecDPDMolecular::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::BOND ) {
        events.push_back(
            transfer_bond(
                direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
}


