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
#include "atom_vec_dpd_angle_meso.h"

using namespace LAMMPS_NS;

#define DELTA 10000

AtomVecDPDAngle::AtomVecDPDAngle( LAMMPS *lmp ) :
    AtomVecAngle( lmp ),
    AtomVecDPDMolecular( lmp ),
    dev_e_bond( lmp, "AtomVecDPDAngle::dev_e_bond" ),
    dev_nbond( lmp, "AtomVecDPDAngle::dev_nbond" ),
    dev_bond( lmp, "AtomVecDPDAngle::dev_bond" ),
    dev_bond_mapped( lmp, "AtomVecDPDAngle::dev_bond_mapped" ),
    dev_e_angle( lmp, "AtomVecDPDAngle::dev_e_angle" ),
    dev_nangle( lmp, "AtomVecDPDAngle::dev_nangle" ),
    dev_angle( lmp, "AtomVecDPDAngle::dev_angle" ),
    dev_angle_mapped( lmp, "AtomVecDPDAngle::dev_angle_mapped" ),
    dev_nbond_pinned( lmp, "AtomVecDPDAngle::dev_nbond_pinned" ),
    dev_bond_atom_pinned( lmp, "AtomVecDPDAngle::dev_bond_atom_pinned" ),
    dev_bond_type_pinned( lmp, "AtomVecDPDAngle::dev_bond_type_pinned" ),
    dev_nangle_pinned( lmp, "AtomVecDPDAngle::dev_nangle_pinned" ),
    dev_angle_atom1_pinned( lmp, "AtomVecDPDAngle::dev_angle_atom1_pinned" ),
    dev_angle_atom2_pinned( lmp, "AtomVecDPDAngle::dev_angle_atom2_pinned" ),
    dev_angle_atom3_pinned( lmp, "AtomVecDPDAngle::dev_angle_atom3_pinned" ),
    dev_angle_type_pinned( lmp, "AtomVecDPDAngle::dev_angle_type_pinned" ),
    hst_angle_id( lmp, "AtomVecDPDAngle::hst_angle_id" ),
    hst_angle_packed( lmp, "AtomVecDPDAngle::hst_angle_packed" ),
    dev_angle_id( lmp, "AtomVecDPDAngle::dev_angle_id" ),
    dev_angle_packed( lmp, "AtomVecDPDAngle::dev_angle_packed" )
{
    cudable       = 1;
    comm_x_only   = 0;

    pre_sort     = AtomAttribute::LOCAL  | AtomAttribute::COORD;
    post_sort    = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE |
                   AtomAttribute::EXCL   | AtomAttribute::BOND     | AtomAttribute::ANGLE ;
    pre_border   = AtomAttribute::BORDER | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    post_border  = AtomAttribute::GHOST  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE;
    pre_comm     = AtomAttribute::BORDER | AtomAttribute::COORD    | AtomAttribute::VELOC;
    post_comm    = AtomAttribute::GHOST  | AtomAttribute::COORD    | AtomAttribute::VELOC;
    pre_exchange = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::MOLE |
                   AtomAttribute::EXCL   | AtomAttribute::BOND     | AtomAttribute::ANGLE;
    pre_output   = AtomAttribute::LOCAL  | AtomAttribute::ESSENTIAL | AtomAttribute::FORCE |
                   AtomAttribute::MOLE   | AtomAttribute::EXCL     | AtomAttribute::BOND  |
                   AtomAttribute::ANGLE;
}

void AtomVecDPDAngle::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecDPDAngle::grow_reset()
{
    AtomVecAngle::grow_reset();
    grow_exclusion();
}

void AtomVecDPDAngle::grow_cpu( int n )
{
    AtomVecAngle::grow( n );
}

void AtomVecDPDAngle::grow_device( int nmax_new )
{
    AtomVecDPDMolecular::grow_device( nmax_new );

    meso_atom->dev_e_bond       = dev_e_bond      .grow( nmax_new, false, true );
    meso_atom->dev_nbond        = dev_nbond       .grow( nmax_new );
    meso_atom->dev_bond        = dev_bond       .grow( nmax_new , atom->bond_per_atom );
    meso_atom->dev_bond_mapped = dev_bond_mapped.grow( nmax_new , atom->bond_per_atom );

    meso_atom->dev_e_angle       = dev_e_angle      .grow( nmax_new, false, true );
    meso_atom->dev_nangle        = dev_nangle       .grow( nmax_new );
    meso_atom->dev_angle        = dev_angle       .grow( nmax_new , atom->angle_per_atom );
    meso_atom->dev_angle_mapped = dev_angle_mapped.grow( nmax_new , atom->angle_per_atom );
}

void AtomVecDPDAngle::pin_host_array()
{
    AtomVecDPDMolecular::pin_host_array();

    if( atom->bond_per_atom ) {
        if( atom->num_bond )  dev_nbond_pinned    .map_host( atom->nmax, atom->num_bond );
        if( atom->bond_atom )  dev_bond_atom_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_atom[0][0] ) );
        if( atom->bond_type )  dev_bond_type_pinned.map_host( atom->nmax * atom->bond_per_atom, &( atom->bond_type[0][0] ) );
    }
    if( atom->angle_per_atom ) {
        if( atom->num_angle ) dev_nangle_pinned     .map_host( atom->nmax, atom->num_angle );
        if( atom->angle_type ) dev_angle_type_pinned .map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_type[0][0] ) );
        if( atom->angle_atom1 ) dev_angle_atom1_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom1[0][0] ) );
        if( atom->angle_atom2 ) dev_angle_atom2_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom2[0][0] ) );
        if( atom->angle_atom3 ) dev_angle_atom3_pinned.map_host( atom->nmax * atom->angle_per_atom, &( atom->angle_atom3[0][0] ) );
    }
}

void AtomVecDPDAngle::unpin_host_array()
{
    AtomVecDPDMolecular::unpin_host_array();

    dev_nbond_pinned    .unmap_host( atom->num_bond );
    dev_bond_atom_pinned.unmap_host( atom->bond_atom ? & ( atom->bond_atom[0][0] ) : NULL );
    dev_bond_type_pinned.unmap_host( atom->bond_type ? & ( atom->bond_type[0][0] ) : NULL );
    dev_nangle_pinned     .unmap_host( atom->num_angle );
    dev_angle_type_pinned .unmap_host( atom->angle_type  ? & ( atom->angle_type[0][0] ) : NULL );
    dev_angle_atom1_pinned.unmap_host( atom->angle_atom1 ? & ( atom->angle_atom1[0][0] ) : NULL );
    dev_angle_atom2_pinned.unmap_host( atom->angle_atom2 ? & ( atom->angle_atom2[0][0] ) : NULL );
    dev_angle_atom3_pinned.unmap_host( atom->angle_atom3 ? & ( atom->angle_atom3[0][0] ) : NULL );
}

void AtomVecDPDAngle::data_atom_target( int i, double *coord, int imagetmp, char **values )
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

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_angle(
    int*  __restrict host_n_angle,
    int*  __restrict host_angle_atom1,
    int*  __restrict host_angle_atom2,
    int*  __restrict host_angle_atom3,
    int*  __restrict host_angle_type,
    int*  __restrict n_angle,
    int4* __restrict angles,
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
            int n = host_n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 angle;
                angle.x = host_angle_atom1[ i * padding_host + p ];
                angle.y = host_angle_atom2[ i * padding_host + p ];
                angle.z = host_angle_atom3[ i * padding_host + p ];
                angle.w = host_angle_type [ i * padding_host + p ];
                angles[ tid + p * padding_device ] = angle;
            }
            n_angle[ tid ] = n;
        } else {
            int n = n_angle[ i ];
            for( int p = 0 ; p < n ; p++ ) {
                int4 angle = angles[ i + p * padding_device ];
                host_angle_atom1[ i * padding_host + p ] = angle.x;
                host_angle_atom2[ i * padding_host + p ] = angle.y;
                host_angle_atom3[ i * padding_host + p ] = angle.z;
                host_angle_type [ i * padding_host + p ] = angle.w;
            }
            host_n_angle[ i ] = n;
        }
    }
}

// DIR = 0 : CPU -> GPU
// DIR = 1 : GPU -> CPU
template<int DIR>
__global__ void gpu_copy_angle_new(
    int*  __restrict host_n_angle,
    int*  __restrict host_angle_atom1,
    int*  __restrict host_angle_atom2,
    int*  __restrict host_angle_atom3,
    int*  __restrict host_angle_type,
    int*  __restrict n_angle,
    int4* __restrict angles,
    int*  __restrict perm_table,
    const int padding_host,
    const double padding_host_inv,
    const int padding_device,
    const int p_beg,
    const int n
)
{
    int hid = blockIdx.x * blockDim.x + threadIdx.x;
//  for(int hid = blockIdx.x * blockDim.x + threadIdx.x; hid < n; hid += gridDim.x * blockDim.x )
    if( hid < n ) {
        int t = hid * padding_host_inv;
        int p = hid - t * padding_host;
        int d = p_beg + t;
        if( DIR == 0 ) {
            int i = d;
            if( perm_table ) i = perm_table[ i ];
            int n = host_n_angle[ i ];
            if( p < n ) {
                int4 angle;
                angle.x = host_angle_atom1[ i * padding_host + p ];
                angle.y = host_angle_atom2[ i * padding_host + p ];
                angle.z = host_angle_atom3[ i * padding_host + p ];
                angle.w = host_angle_type [ i * padding_host + p ];
                angles[ d + p * padding_device ] = angle;
            }
            if( p == 0 ) {
                n_angle[ d ] = n;
            }
        }
    }
}


CUDAEvent AtomVecDPDAngle::transfer_bond( TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action )
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
            dev_nbond,
            dev_bond,
            NULL,
            atom->bond_per_atom,
            dev_bond.pitch(),
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

__global__ void gpu_copy_angle_beta(
		int * tag,
    int2* __restrict atom_angle,
    int4* __restrict hst_angles,
    int4* __restrict dev_angle,
    int*  __restrict perm_to,
    const int padding_device,
    const int n
)
{
    for( int h = blockIdx.x * blockDim.x + threadIdx.x; h < n; h += gridDim.x * blockDim.x ) {
        int2 a = atom_angle[h];
        int d = perm_to ? perm_to[ a.x ] : a.x;
        dev_angle[ d + a.y * padding_device ] = hst_angles[h];
    }
}

void AtomVecDPDAngle::pack_angle( CUDAStream stream )
{
    int n = 0;
    nangle_prefix.clear();
    for( int i = 0; i < atom->nlocal; i++ ) {
        nangle_prefix.push_back( n );
        n += atom->num_angle[i];
    }
    if( dev_angle_id.n() < n ) {
        meso_device->sync_device();
        hst_angle_id.grow( n );
        hst_angle_packed.grow( n );
        dev_angle_id.grow( n );
        dev_angle_packed.grow( n );
    }

    int ** const atom1 = atom->angle_atom1, ** const atom2 = atom->angle_atom2;
    int ** const atom3 = atom->angle_atom3, ** const atype = atom->angle_type;

    static ThreadTuner &o = meso_device->tuner( "AtomVecDPDAngle::pack_angle" );
    size_t ntd = o.bet();
    double t1 = meso_device->get_time_omp();

    if( OMPDEBUG ) printf( "%d %s\n", __LINE__, __FILE__ );
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if( tid < ntd ) {
            int2 work = split_work( 0, atom->nlocal, tid, ntd );
            for( int i = work.x; i < work.y; i++ ) {
                for( int j = 0; j < atom->num_angle[i]; j++ ) {
                    hst_angle_id    [ nangle_prefix[i] + j ] = make_int2( i, j );
                    hst_angle_packed[ nangle_prefix[i] + j ] = make_int4( atom1[i][j], atom2[i][j], atom3[i][j], atype[i][j] );
                }
            }
        }
    }

    double t2 = meso_device->get_time_omp();
    if( meso_device->warmed_up() ) o.learn( ntd, t2 - t1 );

    dev_angle_id    .upload( hst_angle_id,     n, stream );
    dev_angle_packed.upload( hst_angle_packed, n, stream );
}

CUDAEvent AtomVecDPDAngle::transfer_angle( TransferDirection direction, int* permute_to, int p_beg, int n_transfer, CUDAStream stream, int action )
{
    CUDAEvent e;
    if( direction == CUDACOPY_C2G ) {
        // sparse solution
    	// BUGGY!
        /*static GridConfig grid_cfg = meso_device->configure_kernel( gpu_copy_angle_beta , 0 );
        if( action & ACTION_COPY ) {
            pack_angle( stream );
        }
        meso_device->sync_device();
        if( action & ACTION_PERM )
            gpu_copy_angle_beta <<< grid_cfg.x, grid_cfg.y, 0, stream >>>(
            	dev_tag,
                dev_angle_id,
                dev_angle_packed,
                dev_angle,
                permute_to,
                dev_angle.pitch(),
                dev_angle_packed.n() );
        e = meso_device->event( "AtomVecDPDAngle::angle::C2G" );
        e.record( stream );*/

        // dense solution
      if (action & ACTION_COPY) dev_nangle_pinned.upload(n_transfer, stream, p_beg);
      if (action & ACTION_COPY) dev_angle_atom1_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_atom2_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_atom3_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      if (action & ACTION_COPY) dev_angle_type_pinned.upload(n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom);
      int threads_per_block = meso_device->query_block_size(gpu_copy_angle<0>);
      if (action & ACTION_PERM)
          gpu_copy_angle<0> <<< n_block(n_transfer,threads_per_block), threads_per_block, 0, stream >>>(
              dev_nangle_pinned.buf_d(),
              dev_angle_atom1_pinned.buf_d(),
              dev_angle_atom2_pinned.buf_d(),
              dev_angle_atom3_pinned.buf_d(),
              dev_angle_type_pinned.buf_d(),
              dev_nangle,
              dev_angle,
              meso_atom->dev_permute_from,
              atom->angle_per_atom,
              dev_angle.pitch(),
              p_beg,
              n_transfer );
      e = meso_device->event("AtomVecDPDAngle::angle::C2G");
      e.record(stream);

//      if ( update->ntimestep > 10 )
//      {
//          char filename[256];
//          sprintf( filename ,"dev_angle.v%d.%d", version, update->ntimestep );
//          std::std::vector<int > n( dev_nangle.n() );
//          std::std::vector<int4> v( dev_angle.pitch() * dev_angle.h() );
//          std::ofstream fout( filename );
//          verify(( cudaMemcpy( n.data(), dev_nangle.ptr(), n.size() * sizeof(int ), cudaMemcpyDefault ) ));
//          verify(( cudaMemcpy( v.data(), dev_angle.ptr(), v.size() * sizeof(int4), cudaMemcpyDefault ) ));
//          for(int i = 0 ; i < atom->nlocal ; i++) {
//              fout<<n[i]<<":\t";
//              for(int j = 0 ; j < n[i] ; j++) {
//                  fout<< v[ i+j*dev_angle.pitch() ] << '\t';
//              }
//              fout<<endl;
//          }
//          cout<< update->ntimestep <<endl;
//          fast_exit(0);
//      }

    } else {
        int threads_per_block = meso_device->query_block_size( gpu_copy_angle<1> );
        gpu_copy_angle<1> <<< n_block( n_transfer, threads_per_block ), threads_per_block, 0, stream >>>(
            dev_nangle_pinned.buf_d(),
            dev_angle_atom1_pinned.buf_d(),
            dev_angle_atom2_pinned.buf_d(),
            dev_angle_atom3_pinned.buf_d(),
            dev_angle_type_pinned.buf_d(),
            dev_nangle,
            dev_angle,
            NULL,
            atom->angle_per_atom,
            dev_angle.pitch(),
            p_beg,
            n_transfer );
        dev_nangle_pinned.download( n_transfer, stream, p_beg );
        dev_angle_atom1_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom2_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_atom3_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        dev_angle_type_pinned.download( n_transfer * atom->angle_per_atom, stream, p_beg * atom->angle_per_atom );
        e = meso_device->event( "AtomVecDPDAngle::angle::G2C" );
        e.record( stream );
    }
    return e;
}

void AtomVecDPDAngle::transfer_impl(
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
}


