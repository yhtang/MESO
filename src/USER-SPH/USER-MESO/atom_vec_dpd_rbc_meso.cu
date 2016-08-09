#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"
#include "update.h"

#include "meso.h"
#include "atom_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_dpd_rbc_meso.h"

using namespace LAMMPS_NS;

#define DELTA 10000

AtomVecDPDRBC::AtomVecDPDRBC( LAMMPS *lmp ) :
    AtomVecRBC( lmp ),
    AtomVecDPDMolecular( lmp ),
    dev_e_bond( lmp, "AtomVecDPDRBC::dev_e_bond" ),
    dev_nbond( lmp, "AtomVecDPDRBC::dev_nbond" ),
    dev_bond( lmp, "AtomVecDPDRBC::dev_bond" ),
    dev_bond_mapped( lmp, "AtomVecDPDRBC::dev_bond_mapped" ),
    dev_bond_r0( lmp, "AtomVecDPDRBC::dev_bond_r0" ),
    dev_e_angle( lmp, "AtomVecDPDRBC::dev_e_angle" ),
    dev_nangle( lmp, "AtomVecDPDRBC::dev_nangle" ),
    dev_angle( lmp, "AtomVecDPDRBC::dev_angle" ),
    dev_angle_mapped( lmp, "AtomVecDPDRBC::dev_angle_mapped" ),
    dev_angle_a0( lmp, "AtomVecDPDRBC::dev_angle_a0" ),
	dev_e_dihed( lmp, "AtomVecDPDRBC::dev_e_dihed" ),
	dev_ndihed( lmp, "AtomVecDPDRBC::dev_ndihed" ),
	dev_dihed_type( lmp, "AtomVecDPDRBC::dev_dihed_type" ),
	dev_dihed( lmp, "AtomVecDPDRBC::dev_dihed" ),
	dev_dihed_mapped( lmp, "AtomVecDPDRBC::dev_dihed_mapped" ),
    dev_nbond_pinned( lmp, "AtomVecDPDRBC::dev_nbond_pinned" ),
    dev_bond_atom_pinned( lmp, "AtomVecDPDRBC::dev_bond_atom_pinned" ),
    dev_bond_type_pinned( lmp, "AtomVecDPDRBC::dev_bond_type_pinned" ),
    dev_bond_r0_pinned( lmp, "AtomVecDPDRBC::dev_bond_r0_pinned" ),
    dev_nangle_pinned( lmp, "AtomVecDPDRBC::dev_nangle_pinned" ),
    dev_angle_atom1_pinned( lmp, "AtomVecDPDRBC::dev_angle_atom1_pinned" ),
    dev_angle_atom2_pinned( lmp, "AtomVecDPDRBC::dev_angle_atom2_pinned" ),
    dev_angle_atom3_pinned( lmp, "AtomVecDPDRBC::dev_angle_atom3_pinned" ),
    dev_angle_type_pinned( lmp, "AtomVecDPDRBC::dev_angle_type_pinned" ),
    dev_angle_a0_pinned( lmp, "AtomVecDPDRBC::dev_angle_a0_pinned" ),
	dev_ndihed_pinned( lmp, "AtomVecDPDRBC::dev_ndihed_pinned" ),
	dev_dihed_atom1_pinned( lmp, "AtomVecDPDRBC::dev_dihed_atom1_pinned" ),
	dev_dihed_atom2_pinned( lmp, "AtomVecDPDRBC::dev_dihed_atom2_pinned" ),
	dev_dihed_atom3_pinned( lmp, "AtomVecDPDRBC::dev_dihed_atom3_pinned" ),
	dev_dihed_atom4_pinned( lmp, "AtomVecDPDRBC::dev_dihed_atom4_pinned" ),
	dev_dihed_type_pinned( lmp, "AtomVecDPDRBC::dev_dihed_type_pinned" )
{
    cudable       = 1;
    comm_x_only   = 0;

    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  |
                   AtomAttribute::EXCL   | AtomAttribute::BOND      | AtomAttribute::ANGLE |
                   AtomAttribute::DIHED ;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD     | AtomAttribute::VELOC;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD     | AtomAttribute::VELOC;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE  |
                   AtomAttribute::EXCL   | AtomAttribute::BOND      | AtomAttribute::ANGLE |
                   AtomAttribute::DIHED;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE |
                   AtomAttribute::MOLE   | AtomAttribute::EXCL      | AtomAttribute::BOND  |
                   AtomAttribute::ANGLE  | AtomAttribute::DIHED;
}

void AtomVecDPDRBC::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecDPDRBC::grow_reset()
{
    AtomVecRBC::grow_reset();
    grow_exclusion();
}

void AtomVecDPDRBC::grow_cpu( int n )
{
    AtomVecRBC::grow( n );
}

void AtomVecDPDRBC::grow_device( int nmax_new )
{
    AtomVecDPDMolecular::grow_device( nmax_new );

    meso_atom->dev_e_bond       = dev_e_bond     .grow( nmax_new, false, true );
    meso_atom->dev_nbond        = dev_nbond      .grow( nmax_new );
    meso_atom->dev_bond         = dev_bond       .grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_mapped  = dev_bond_mapped.grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_r0      = dev_bond_r0    .grow( nmax_new , atom->bond_per_atom );

    meso_atom->dev_e_angle      = dev_e_angle      .grow( nmax_new, false, true );
    meso_atom->dev_nangle       = dev_nangle       .grow( nmax_new );
    meso_atom->dev_angle        = dev_angle       .grow( nmax_new , atom->angle_per_atom );
    meso_atom->dev_angle_mapped = dev_angle_mapped.grow( nmax_new , atom->angle_per_atom );
    meso_atom->dev_angle_a0     = dev_angle_a0    .grow( nmax_new , atom->angle_per_atom );

    meso_atom->dev_e_dihed       = dev_e_dihed      .grow( nmax_new, false, true );
    meso_atom->dev_ndihed        = dev_ndihed       .grow( nmax_new );
    meso_atom->dev_dihed_type    = dev_dihed_type   .grow( nmax_new , atom->dihedral_per_atom );
    meso_atom->dev_dihed        = dev_dihed       .grow( nmax_new , atom->dihedral_per_atom );
    meso_atom->dev_dihed_mapped = dev_dihed_mapped.grow( nmax_new , atom->dihedral_per_atom );
}

void AtomVecDPDRBC::pin_host_array()
{
    AtomVecDPDMolecular::pin_host_array();

    if( atom->bond_per_atom ) {
        if( atom->num_bond  )  dev_nbond_pinned    .map_host( atom->nmax, atom->num_bond );
        if( atom->bond_atom )  dev_bond_atom_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_atom[0][0] ) );
        if( atom->bond_type )  dev_bond_type_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_type[0][0] ) );
        if( atom->bond_r0   )  dev_bond_r0_pinned  .map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_r0  [0][0] ) );
    }
    if( atom->angle_per_atom ) {
        if( atom->num_angle   ) dev_nangle_pinned     .map_host( atom->nmax, atom->num_angle );
        if( atom->angle_type  ) dev_angle_type_pinned .map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_type[0][0] ) );
        if( atom->angle_atom1 ) dev_angle_atom1_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom1[0][0] ) );
        if( atom->angle_atom2 ) dev_angle_atom2_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom2[0][0] ) );
        if( atom->angle_atom3 ) dev_angle_atom3_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom3[0][0] ) );
        if( atom->angle_a0    ) dev_angle_a0_pinned   .map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_a0   [0][0] ) );
    }
    if( atom->dihedral_per_atom ) {
        if( atom->num_dihedral   ) dev_ndihed_pinned     .map_host( atom->nmax, atom->num_dihedral );
        if( atom->dihedral_type  ) dev_dihed_type_pinned .map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_type[0][0] ) );
        if( atom->dihedral_atom1 ) dev_dihed_atom1_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom1[0][0] ) );
        if( atom->dihedral_atom2 ) dev_dihed_atom2_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom2[0][0] ) );
        if( atom->dihedral_atom3 ) dev_dihed_atom3_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom3[0][0] ) );
        if( atom->dihedral_atom4 ) dev_dihed_atom4_pinned.map_host( atom->nmax * atom->dihedral_per_atom, &( atom->dihedral_atom4[0][0] ) );
    }
}

void AtomVecDPDRBC::unpin_host_array()
{
    AtomVecDPDMolecular::unpin_host_array();

    dev_nbond_pinned    .unmap_host( atom->num_bond );
    dev_bond_atom_pinned.unmap_host( atom->bond_atom ? & ( atom->bond_atom[0][0] ) : NULL );
    dev_bond_type_pinned.unmap_host( atom->bond_type ? & ( atom->bond_type[0][0] ) : NULL );
    dev_bond_r0_pinned  .unmap_host( atom->bond_r0   ? & ( atom->bond_r0  [0][0] ) : NULL );
    dev_nangle_pinned     .unmap_host( atom->num_angle );
    dev_angle_type_pinned .unmap_host( atom->angle_type  ? & ( atom->angle_type[0][0] ) : NULL );
    dev_angle_atom1_pinned.unmap_host( atom->angle_atom1 ? & ( atom->angle_atom1[0][0] ) : NULL );
    dev_angle_atom2_pinned.unmap_host( atom->angle_atom2 ? & ( atom->angle_atom2[0][0] ) : NULL );
    dev_angle_atom3_pinned.unmap_host( atom->angle_atom3 ? & ( atom->angle_atom3[0][0] ) : NULL );
    dev_angle_a0_pinned   .unmap_host( atom->angle_a0    ? & ( atom->angle_a0   [0][0] ) : NULL );
    dev_ndihed_pinned     .unmap_host( atom->num_dihedral );
    dev_dihed_type_pinned .unmap_host( atom->dihedral_type  ? & ( atom->dihedral_type[0][0] ) : NULL );
    dev_dihed_atom1_pinned.unmap_host( atom->dihedral_atom1 ? & ( atom->dihedral_atom1[0][0] ) : NULL );
    dev_dihed_atom2_pinned.unmap_host( atom->dihedral_atom2 ? & ( atom->dihedral_atom2[0][0] ) : NULL );
    dev_dihed_atom3_pinned.unmap_host( atom->dihedral_atom3 ? & ( atom->dihedral_atom3[0][0] ) : NULL );
    dev_dihed_atom4_pinned.unmap_host( atom->dihedral_atom4 ? & ( atom->dihedral_atom4[0][0] ) : NULL );
}

void AtomVecDPDRBC::data_atom_target( int i, double *coord, int imagetmp, char **values )
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
    r64*  __restrict host_bond_r0,
    int*  __restrict n_bond,
    int2* __restrict bond,
    r64*  __restrict bond_r0,
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
                int2 b;
                b.x = host_bond_atom[ i * padding_host + p ];
                b.y = host_bond_type[ i * padding_host + p ];
                bond[ tid + p * padding_device ] = b;
                bond_r0[ tid + p * padding_device ] = host_bond_r0[ i * padding_host + p ];
            }
            n_bond[ tid ] = n;
        } else {
            int n = n_bond[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int2 b = bond[ i + p * padding_device ];
                host_bond_atom[ i * padding_host + p ] = b.x;
                host_bond_type[ i * padding_host + p ] = b.y;
                host_bond_r0[ i * padding_host + p ] = bond_r0[ i + p * padding_device ];
            }
            host_n_bond[ i ] = n;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_angle(
    int*  __restrict host_n_angle,
    int*  __restrict host_angle_atom1,
    int*  __restrict host_angle_atom2,
    int*  __restrict host_angle_atom3,
    int*  __restrict host_angle_type,
    r64*  __restrict host_angle_a0,
    int*  __restrict n_angle,
    int4* __restrict angle,
    r64*  __restrict angle_a0,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int padding_device_a0,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 a;
                a.x = host_angle_atom1[ i * padding_host + p ];
                a.y = host_angle_atom2[ i * padding_host + p ];
                a.z = host_angle_atom3[ i * padding_host + p ];
                a.w = host_angle_type [ i * padding_host + p ];
                angle[ tid + p * padding_device ] = a;
                angle_a0[ tid + p * padding_device_a0 ] = host_angle_a0[ i * padding_host + p ];
            }
            n_angle[ tid ] = n;
        } else {
            int n = n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 a = angle[ i + p * padding_device ];
                host_angle_atom1[ i * padding_host + p ] = a.x;
                host_angle_atom2[ i * padding_host + p ] = a.y;
                host_angle_atom3[ i * padding_host + p ] = a.z;
                host_angle_type [ i * padding_host + p ] = a.w;
                host_angle_a0[ i * padding_host + p ] = angle_a0[ i + p * padding_device_a0 ];
            }
            host_n_angle[ i ] = n;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_dihed(
    int*  __restrict host_n_dihed,
    int*  __restrict host_dihed_atom1,
    int*  __restrict host_dihed_atom2,
    int*  __restrict host_dihed_atom3,
    int*  __restrict host_dihed_atom4,
    int*  __restrict host_dihed_type,
    int*  __restrict n_dihed,
    int*  __restrict dihed_type,
    int4* __restrict diheds,
    int*  __restrict perm_table,
    const int padding_host,
    const int padding_device,
    const int padding_device_type,
    const int p_beg,
    const int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < n ) {
        int i = p_beg + tid;
        if( DIR == 0 ) {
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_dihed[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 dihed;
                dihed.x = host_dihed_atom1[ i * padding_host + p ];
                dihed.y = host_dihed_atom2[ i * padding_host + p ];
                dihed.z = host_dihed_atom3[ i * padding_host + p ];
                dihed.w = host_dihed_atom4[ i * padding_host + p ];
                diheds[ tid + p * padding_device ] = dihed;
                dihed_type[ tid + p * padding_device_type ] = host_dihed_type[ i * padding_host + p ];
            }
            n_dihed[ tid ] = n;
        } else {
            int n = n_dihed[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 dihed = diheds[ i + p * padding_device ];
                host_dihed_atom1[ i * padding_host + p ] = dihed.x;
                host_dihed_atom2[ i * padding_host + p ] = dihed.y;
                host_dihed_atom3[ i * padding_host + p ] = dihed.z;
                host_dihed_atom4[ i * padding_host + p ] = dihed.w;
                host_dihed_type [ i * padding_host + p ] = dihed_type[ i + p * padding_device_type ];
            }
            host_n_dihed[ i ] = n;
        }
    }
}

CUDAEvent AtomVecDPDRBC::transfer_bond( TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        if( action & ACTION_COPY ) dev_nbond_pinned.upload( n_transfer, stream, p_beg );
        if( action & ACTION_COPY ) dev_bond_atom_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        if( action & ACTION_COPY ) dev_bond_type_pinned.upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        if( action & ACTION_COPY ) dev_bond_r0_pinned  .upload( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        int threads_per_block = meso_device->query_block_size( gpu_copy_bond<0> );
        if( action & ACTION_PERM )
            gpu_copy_bond<0> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
                dev_nbond_pinned.buf_d(),
                dev_bond_atom_pinned.buf_d(),
                dev_bond_type_pinned.buf_d(),
                dev_bond_r0_pinned.buf_d(),
                dev_nbond,
                dev_bond,
                dev_bond_r0,
                permute_from,
                atom->bond_per_atom,
                dev_bond.pitch(),
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
            dev_bond_r0_pinned.buf_d(),
            dev_nbond,
            dev_bond,
            dev_bond_r0,
            NULL,
            atom->bond_per_atom,
            dev_bond.pitch(),
            p_beg,
            n_transfer );
        dev_nbond_pinned.download( n_transfer, stream, p_beg );
        dev_bond_atom_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        dev_bond_type_pinned.download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        dev_bond_r0_pinned  .download( n_transfer * atom->bond_per_atom, stream, p_beg * atom->bond_per_atom );
        e = meso_device->event( "AtomVecDPDBond::transfer_bond::G2C" );
        e.record( stream );
    }
    return e;
}

CUDAEvent AtomVecDPDRBC::transfer_angle( TransferDirection direction, int* permute_to, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
      if (action & ACTION_COPY) dev_nangle_pinned.upload(n_transfer, stream, p_beg);
      if (action & ACTION_COPY) dev_angle_atom1_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_atom2_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_atom3_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_type_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_a0_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      int threads_per_block = meso_device->query_block_size(gpu_copy_angle<0>);
      if (action & ACTION_PERM)
          gpu_copy_angle<0> <<< n_block(n_transfer,threads_per_block), threads_per_block, 0, stream >>>(
              dev_nangle_pinned.buf_d(),
              dev_angle_atom1_pinned.buf_d(),
              dev_angle_atom2_pinned.buf_d(),
              dev_angle_atom3_pinned.buf_d(),
              dev_angle_type_pinned.buf_d(),
              dev_angle_a0_pinned.buf_d(),
              dev_nangle,
              dev_angle,
              dev_angle_a0,
              meso_atom->dev_permute_from,
              atom->angle_per_atom,
              dev_angle.pitch(),
              dev_angle_a0.pitch(),
              p_beg,
              n_transfer );
      e = meso_device->event("AtomVecDPDRBC::angle::C2G");
      e.record(stream);
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_angle<1> );
        gpu_copy_angle<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_nangle_pinned.buf_d(),
            dev_angle_atom1_pinned.buf_d(),
            dev_angle_atom2_pinned.buf_d(),
            dev_angle_atom3_pinned.buf_d(),
            dev_angle_type_pinned.buf_d(),
            dev_angle_a0_pinned.buf_d(),
            dev_nangle,
            dev_angle,
            dev_angle_a0,
            NULL,
            atom->angle_per_atom,
            dev_angle.pitch(),
            dev_angle_a0.pitch(),
            p_beg,
            n_transfer );
        dev_nangle_pinned.download( n_transfer, stream, p_beg );
        dev_angle_atom1_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom2_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom3_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_type_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_a0_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        e = meso_device->event( "AtomVecDPDRBC::angle::G2C" );
        e.record( stream );
    }
    return e;
}

CUDAEvent AtomVecDPDRBC::transfer_dihed( TransferDirection direction, int* permute_to, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
      if (action & ACTION_COPY) dev_ndihed_pinned.upload(n_transfer, stream, p_beg);
      if (action & ACTION_COPY) dev_dihed_atom1_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
      if (action & ACTION_COPY) dev_dihed_atom2_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
      if (action & ACTION_COPY) dev_dihed_atom3_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
      if (action & ACTION_COPY) dev_dihed_atom4_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
      if (action & ACTION_COPY) dev_dihed_type_pinned.upload(n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom);
      int threads_per_block = meso_device->query_block_size(gpu_copy_dihed<0>);
      if (action & ACTION_PERM)
          gpu_copy_dihed<0> <<< n_block(n_transfer,threads_per_block), threads_per_block, 0, stream >>>(
			  dev_ndihed_pinned.buf_d(),
              dev_dihed_atom1_pinned.buf_d(),
              dev_dihed_atom2_pinned.buf_d(),
              dev_dihed_atom3_pinned.buf_d(),
              dev_dihed_atom4_pinned.buf_d(),
              dev_dihed_type_pinned.buf_d(),
              dev_ndihed,
              dev_dihed_type,
              dev_dihed,
              meso_atom->dev_permute_from,
              atom->dihedral_per_atom,
              dev_dihed.pitch(),
              dev_dihed_type.pitch(),
              p_beg,
              n_transfer );
      e = meso_device->event("AtomVecDPDRBC::dihed::C2G");
      e.record(stream);
    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_dihed<1> );
        gpu_copy_dihed<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_ndihed_pinned.buf_d(),
            dev_dihed_atom1_pinned.buf_d(),
            dev_dihed_atom2_pinned.buf_d(),
            dev_dihed_atom3_pinned.buf_d(),
            dev_dihed_atom4_pinned.buf_d(),
            dev_dihed_type_pinned.buf_d(),
            dev_ndihed,
            dev_dihed_type,
            dev_dihed,
            NULL,
            atom->dihedral_per_atom,
            dev_dihed.pitch(),
            dev_dihed_type.pitch(),
            p_beg,
            n_transfer );
        dev_ndihed_pinned.download( n_transfer, stream, p_beg );
        dev_dihed_atom1_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom2_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom3_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_atom4_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        dev_dihed_type_pinned.download( n_transfer * atom->dihedral_per_atom, stream, p_beg * atom->dihedral_per_atom );
        e = meso_device->event( "AtomVecDPDRBC::dihed::G2C" );
        e.record( stream );
    }
    return e;
}

void AtomVecDPDRBC::transfer_impl(
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
    if( per_atom_prop & AtomAttribute::ANGLE ) {
        events.push_back(
            transfer_scalar(
                dev_nangle_pinned, dev_nangle, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
        events.push_back(
            transfer_angle(
                direction, permute_to, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
    if( per_atom_prop & AtomAttribute::DIHED ) {
        events.push_back(
            transfer_scalar(
                dev_ndihed_pinned, dev_ndihed, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
        events.push_back(
            transfer_dihed(
                direction, permute_to, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
}


