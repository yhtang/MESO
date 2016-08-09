#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "limits.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "group.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "update.h"
#include "respa.h"
#include "output.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "neigh_request.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "bin_meso.h"
#include "neigh_list_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

#define RQDELTA 1
#define EXDELTA 1

#define LB_FACTOR 1.5
#define SMALL 1.0e-6
#define BIG 1.0e20
#define CUT2BIN_RATIO 100

enum {NSQ, BIN, MULTI};      // also in neigh_list.cpp

//texture<int,  cudaTextureType1D, cudaReadModeElementType> tex_map;
//texture<uint, cudaTextureType1D, cudaReadModeElementType> tex_hash;

MesoNeighbor::MesoNeighbor( LAMMPS *lmp ) : Neighbor( lmp ), MesoPointers( lmp ), cuda_bin( lmp ), sorter( lmp ),
    max_local_bin_size( lmp, "MesoNeighbor::max_local_bin_size" )
{
    max_local_bin_size.grow( 1 );
}

MesoNeighbor::~MesoNeighbor()
{
    for( dlist_iter i = lists_device.begin(); i != lists_device.end(); i++ ) {
        delete i->second;
    }
}

void MesoNeighbor::init()
{
    for( int i = 0; i < nrequest; i++ ) {
        if( requests[i]->cudable ) {
        	if ( lists_device.find(i) != lists_device.end() ) delete lists_device[i];
            lists_device[i] = new MesoNeighList( lmp );
            lists_device[i]->index = i;
        }
    }

    Neighbor::init();
}

template<int HASH_MAX_LOOKUP>
inline __device__ uint hash_lookup( texobj tex_hash_key, int tag, uint nonce, const int hash_table_size )
{
    uint p;
    uint target = tag;
    int c = 0;
    do {
        __TEA_core<8>( target, nonce );
        p = target % hash_table_size;
    } while( tex1Dfetch<uint>( tex_hash_key, p ) != tag && c++ < HASH_MAX_LOOKUP );
    if( c >= HASH_MAX_LOOKUP ) {
        printf( "<MESO> particle %d cannot find neighbor\n", tag );
    }
    return p;
}

__global__ void gpu_map_bond(
    texobj tex_map_array,
    int* __restrict nbond,
    int2* __restrict bonds,
    int2* __restrict bonds_mapped,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_bond = nbond[i];
        for( int p = 0 ; p < n_bond ; p++ ) {
            int2 bond = bonds[ i + p * padding ];
            bond.x = tex1Dfetch<int>( tex_map_array, bond.x );
            bonds_mapped[ i + p * padding ] = bond;
        }
    }
}

__global__ void gpu_map_bond_hash(
    texobj tex_hash_key,
    texobj tex_hash_val,
    int* __restrict tag,
    int* __restrict nbond,
    int2* __restrict bonds,
    int2* __restrict bonds_mapped,
    const uint nonce,
    const int hash_table_size,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_bond = nbond[i];
        for( int j = 0 ; j < n_bond ; j++ ) {
            int2 bond = bonds[ i + j * padding ];
            bond.x = tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, bond.x, nonce, hash_table_size ) );
            bonds_mapped[ i + j * padding ] = bond;
        }
    }
}

void MesoNeighbor::bond_all()
{
#ifdef LMP_MESO_LOG_L2
    fprintf( stderr, "<MESO> Rebuilding bond table on device %d\n", meso_device->DevicePool[0] );
#endif

    if( atom->map_style == 1 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_bond );
        gpu_map_bond <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_map_array,
            meso_atom->dev_nbond,
            meso_atom->dev_bond,
            meso_atom->dev_bond_mapped,
            meso_atom->dev_bond_mapped.pitch(),
            atom->nlocal );
    } else if( atom->map_style == 2 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_bond_hash );
        gpu_map_bond_hash <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_hash_key,
            meso_atom->tex_hash_val,
            meso_atom->dev_tag,
            meso_atom->dev_nbond,
            meso_atom->dev_bond,
            meso_atom->dev_bond_mapped,
            meso_atom->nonce,
            meso_atom->hash_table_size,
            meso_atom->dev_bond_mapped.pitch(),
            atom->nlocal );
    }
}

__global__ void gpu_map_angle(
    texobj tex_map_array,
    int*  __restrict tag,
    int*  __restrict nangle,
    int4* __restrict angles,
    int4* __restrict angles_mapped,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_angle = nangle[i];
        for( int p = 0 ; p < n_angle ; p++ ) {
            int4 angle = angles[ i + p * padding ];
            angle.x = ( tag[i] == angle.x ) ? i : tex1Dfetch<int>( tex_map_array, angle.x );
            angle.y = ( tag[i] == angle.y ) ? i : tex1Dfetch<int>( tex_map_array, angle.y );
            angle.z = ( tag[i] == angle.z ) ? i : tex1Dfetch<int>( tex_map_array, angle.z );
            angles_mapped[ i + p * padding ] = angle;
        }
    }
}

template<int HASH_MAX_LOOKUP>
__global__ void gpu_map_angle_hash(
    texobj tex_hash_key,
    texobj tex_hash_val,
    int*  __restrict tag,
    int*  __restrict nangle,
    int4* __restrict angles,
    int4* __restrict angles_mapped,
    const uint nonce,
    const int hash_table_size,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_angle = nangle[i];
        for( int j = 0 ; j < n_angle ; j++ ) {
            int4 angle = angles[ i + j * padding ];
            angle.x = ( tag[i] == angle.x ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, angle.x, nonce, hash_table_size ) );
            angle.y = ( tag[i] == angle.y ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, angle.y, nonce, hash_table_size ) );
            angle.z = ( tag[i] == angle.z ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, angle.z, nonce, hash_table_size ) );
            angles_mapped[ i + j * padding ] = angle;
        }
    }
}

void MesoNeighbor::angle_all()
{
    if( atom->map_style == 1 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_angle );
        gpu_map_angle <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_map_array,
            meso_atom->dev_tag,
            meso_atom->dev_nangle,
            meso_atom->dev_angle,
            meso_atom->dev_angle_mapped,
            meso_atom->dev_angle_mapped.pitch(),
            atom->nlocal );
    } else if( atom->map_style == 2 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_angle_hash<256> );
        gpu_map_angle_hash<256> <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_hash_key,
            meso_atom->tex_hash_val,
            meso_atom->dev_tag,
            meso_atom->dev_nangle,
            meso_atom->dev_angle,
            meso_atom->dev_angle_mapped,
            meso_atom->nonce,
            meso_atom->hash_table_size,
            meso_atom->dev_angle_mapped.pitch(),
            atom->nlocal );
    }
}

__global__ void gpu_map_dihed(
    texobj tex_map_array,
    int*  __restrict tag,
    int*  __restrict ndihed,
    int4* __restrict diheds,
    int4* __restrict diheds_mapped,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_dihed = ndihed[i];
        for( int p = 0 ; p < n_dihed ; p++ ) {
            int4 dihed = diheds[ i + p * padding ], mapped;
            mapped.x = ( tag[i] == dihed.x ) ? i : tex1Dfetch<int>( tex_map_array, dihed.x );
            mapped.y = ( tag[i] == dihed.y ) ? i : tex1Dfetch<int>( tex_map_array, dihed.y );
            mapped.z = ( tag[i] == dihed.z ) ? i : tex1Dfetch<int>( tex_map_array, dihed.z );
            mapped.w = ( tag[i] == dihed.w ) ? i : tex1Dfetch<int>( tex_map_array, dihed.w );
            diheds_mapped[ i + p * padding ] = mapped;
        }
    }
}

template<int HASH_MAX_LOOKUP>
__global__ void gpu_map_dihed_hash(
    texobj tex_hash_key,
    texobj tex_hash_val,
    int*  __restrict tag,
    int*  __restrict ndihed,
    int4* __restrict diheds,
    int4* __restrict diheds_mapped,
    const uint nonce,
    const int hash_table_size,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_dihed = ndihed[i];
        for( int j = 0 ; j < n_dihed ; j++ ) {
            int4 dihed = diheds[ i + j * padding ], mapped;
            mapped.x = ( tag[i] == dihed.x ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, dihed.x, nonce, hash_table_size ) );
            mapped.y = ( tag[i] == dihed.y ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, dihed.y, nonce, hash_table_size ) );
            mapped.z = ( tag[i] == dihed.z ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, dihed.z, nonce, hash_table_size ) );
            mapped.w = ( tag[i] == dihed.w ) ? i : tex1Dfetch<uint>( tex_hash_val, hash_lookup<256>( tex_hash_key, dihed.w, nonce, hash_table_size ) );
            diheds_mapped[ i + j * padding ] = mapped;
        }
    }
}

__global__ void gpu_dihed_sentinel(
    int*  __restrict tag,
    int*  __restrict ndihed,
    int4* __restrict diheds,
    int4* __restrict diheds_mapped,
    const int padding,
    const int n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n_atom ) {
        int n_dihed = ndihed[i];
        for( int p = 0 ; p < n_dihed ; p++ ) {
            int4 dihed = diheds_mapped[ i + p * padding ];
            if ( dihed.x == -1 || dihed.y == -1 || dihed.z == -1 || dihed.w == -1 ) {
            	int4 raw = diheds[ i + p * padding ];
            	printf("<MESO> Dihedral map failure: (%d,%d,%d,%d) => (%d,%d,%d,%d), check if system exploded or try to make ghost zone thicker\n", raw.x, raw.y, raw.z, raw.w, dihed.x, dihed.y, dihed.z, dihed.w );
            }
        }
    }
}

void MesoNeighbor::dihedral_all()
{
    if( atom->map_style == 1 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_dihed );
        gpu_map_dihed <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_map_array,
            meso_atom->dev_tag,
            meso_atom->dev_ndihed,
            meso_atom->dev_dihed,
            meso_atom->dev_dihed_mapped,
            meso_atom->dev_dihed_mapped.pitch(),
            atom->nlocal );
    } else if( atom->map_style == 2 ) {
        size_t threads_per_block = meso_device->query_block_size( gpu_map_dihed_hash<256> );
        gpu_map_dihed_hash<256> <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            meso_atom->tex_hash_key,
            meso_atom->tex_hash_val,
            meso_atom->dev_tag,
            meso_atom->dev_ndihed,
            meso_atom->dev_dihed,
            meso_atom->dev_dihed_mapped,
            meso_atom->nonce,
            meso_atom->hash_table_size,
            meso_atom->dev_dihed_mapped.pitch(),
            atom->nlocal );
    }

    size_t threads_per_block = meso_device->query_block_size( gpu_dihed_sentinel );
    gpu_dihed_sentinel <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        meso_atom->dev_tag,
        meso_atom->dev_ndihed,
        meso_atom->dev_dihed,
        meso_atom->dev_dihed_mapped,
        meso_atom->dev_dihed_mapped.pitch(),
        atom->nlocal );
}

void MesoNeighbor::bond_partial()
{
    error->all( FLERR, "<MESO> Partial build not implemented in USER-MESO." );
}

void MesoNeighbor::angle_partial()
{
    error->all( FLERR, "<MESO> Partial build not implemented in USER-MESO." );
}

void MesoNeighbor::dihedral_partial()
{
    error->all( FLERR, "<MESO> Partial build not implemented in USER-MESO." );
}

void MesoNeighbor::choose_build( int index, NeighRequest *rq )
{
    if( rq ->cudable ) {
        if( rq->ghost )
            pair_build[index] = ( PairPtr )( &MesoNeighbor::full_bin_meso_ghost );
        else
            pair_build[index] = ( PairPtr )( &MesoNeighbor::full_bin_meso );
    } else {
        Neighbor::choose_build( index, rq );
    }
}

void MesoNeighbor::choose_stencil( int index, NeighRequest *rq )
{
    if( rq ->cudable ) {
        stencil_create[index] = ( StencilPtr )( &MesoNeighbor::stencil_full_bin_3d_meso );
    } else {
        Neighbor::choose_stencil( index, rq );
    }
}

__global__ void gpu_assign_bin_id(
    r64       *coord_x,
    r64       *coord_y,
    r64       *coord_z,
    uint      *bin_id,
    int       *atom_id,
    const double3 mybox_lo,
    const double3 mybox_hi,
    const int  mbinx,
    const int  mbiny,
    const int  mbinz,
    const r64  bininvx,
    const r64  bininvy,
    const r64  bininvz,
    const int  n_local,
    const int  n_atom
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i >= n_atom ) return;

    r64 x = coord_x[i];
    r64 y = coord_y[i];
    r64 z = coord_z[i];
    int bid_x = clamp( ( x - mybox_lo.x ) * bininvx + 1.0, 0, mbinx ) ;
    int bid_y = clamp( ( y - mybox_lo.y ) * bininvy + 1.0, 0, mbiny ) ;
    int bid_z = clamp( ( z - mybox_lo.z ) * bininvz + 1.0, 0, mbinz ) ;
    if( i >= n_local ) {
        bid_x = ( x >= mybox_lo.x ) ? ( x <= mybox_hi.x ? bid_x : mbinx - 1 ) : 0 ;
        bid_y = ( y >= mybox_lo.y ) ? ( y <= mybox_hi.y ? bid_y : mbiny - 1 ) : 0 ;
        bid_z = ( z >= mybox_lo.z ) ? ( z <= mybox_hi.z ? bid_z : mbinz - 1 ) : 0 ;
    }

    atom_id[i] = i;
    bin_id[i] = bid_x + mbinx * ( bid_y + bid_z * mbiny );
}

__global__ void gpu_find_bin_boundary(
    uint      *bin_id,
    int       *atom_id,
    int       *bin_location,
    int       *bin_is_ghost,
    const int  n_bin,
    const int  n_local,
    const int  n_atom
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if( tid >= n_atom ) return;

    uint bid_this, bid_left;
    int  ghost_this;

    bid_this   = bin_id[ tid ];
    ghost_this = ( atom_id[ tid ] >= n_local );

    // if a bin is empty, then it is not important if it is ghost or not
    if( tid == 0 ) {
        for( int i = 0 ; i <= bid_this ; i++ )
            bin_location   [ i ] = 0,
                                   bin_is_ghost[ i ] = ghost_this ;
    } else {
        bid_left   = bin_id[ tid - 1 ];
        if( bid_this != bid_left ) {
            for( int i = bid_left + 1 ; i <= bid_this ; i++ )
                bin_location   [ i ] = tid ,
                                       bin_is_ghost[ i ] = ghost_this ;
        }
    }

    if( tid == n_atom - 1 )
        for( int i = bid_this + 1 ; i <= n_bin ; i++ )
            bin_location[ i ] = n_atom,
                                bin_is_ghost[i] = ghost_this;
}

__global__ void gpu_calc_bin_size(
    int *bin_location,
    int *bin_is_ghost,
    int *bin_size,
    int *bin_size_local,
    int  n_bin
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i >= n_bin ) return;

    bin_size     [i] = bin_location[i + 1] - bin_location[i];
    bin_size_local[i] = ( bin_is_ghost[i] ) ? ( 0 ) : ( bin_size[i] ) ;
}

__global__ void gpu_expand_stencil(
    int  *atom_id,
    int  *bin_location,
    int  *bin_size,
    int  *neighbor_count,
    int  *neighbor_bin,
    int  *stencil,
    int  *stencil_len,
    const int neighbin_padding,
    const int stencil_padding,
    const int tuple_size,
    const int n_bin
)
{
    int bin_id = ( blockIdx.x * blockDim.x + threadIdx.x ) / tuple_size ;
    if( bin_id >= n_bin || !bin_size[ bin_id ] ) return;

    int  p_stencil = 0;
    int* target_stencil = stencil + bin_id * stencil_padding;
    for( int i = 0 ; i < neighbor_count[ bin_id ] ; i++ ) {
        int neigh = neighbor_bin[ bin_id * neighbin_padding + i  ] ;
        int beg   = bin_location[ neigh ];
        int size  = bin_location[ neigh + 1 ] - beg ;
        for( int p = threadIdx.x & ( tuple_size - 1 ) ; p < size ; p += tuple_size ) {
#ifdef LMP_MESO_DEBUG
            assert( p_stencil + p < stencil_padding );
#endif
            target_stencil[ p_stencil + p ] = __ldg( atom_id + beg + p );
        }
        p_stencil += size;
    }

    if( ( threadIdx.x & ( tuple_size - 1 ) ) == 0 ) stencil_len[ bin_id ] = p_stencil;
}

__global__ void gpu_sentinel(
    int  *stencil_len,
    int  *bin_size_local,
    const int stencil_padding,
    const int n_bin )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x ; i < n_bin ; i += gridDim.x * blockDim.x ) {
        if( !bin_size_local[ i ] ) continue;
        if( stencil_len[i] > stencil_padding )
            printf( "STENCIL OVERFLOW: %d > %d\n", stencil_len[i], stencil_padding );
    }
}

__global__ void print_bin(
		uint *bin_id,
		int *atm_id,
		const int n )
{
	for(int i=0;i<n;i++) {
		printf("%u %d\n",bin_id[i],atm_id[i]);
	}
}

void MesoNeighbor::binning_meso( MesoNeighList *list, bool ghost )
{
    cuda_bin.alloc_bins();

    int n_atom = atom->nlocal + atom->nghost;
    int n_bin = mbinx * mbiny * mbinz;
    int bin_id_width = ceil( log2( ( double )n_bin ) );
    assert( bin_id_width < 32 );

#if 0
    // compute bin size distribution on CPU
    // for debugging
    std::vector<int> bin_size( mbinx * mbiny * mbinz, 0 );
    for(int i=0;i<atom->nlocal;i++) {

        r64 x = atom->x[i][0];
        r64 y = atom->x[i][1];
        r64 z = atom->x[i][2];
        int bid_x = clamp( ( x - my_box.lo.x ) * bininvx + 1.0, 0, mbinx ) ;
        int bid_y = clamp( ( y - my_box.lo.y ) * bininvy + 1.0, 0, mbiny ) ;
        int bid_z = clamp( ( z - my_box.lo.z ) * bininvz + 1.0, 0, mbinz ) ;
        if( i >= atom->nlocal ) {
            bid_x = ( x >= my_box.lo.x ) ? ( x <= my_box.hi.x ? bid_x : mbinx - 1 ) : 0 ;
            bid_y = ( y >= my_box.lo.y ) ? ( y <= my_box.hi.y ? bid_y : mbiny - 1 ) : 0 ;
            bid_z = ( z >= my_box.lo.z ) ? ( z <= my_box.hi.z ? bid_z : mbinz - 1 ) : 0 ;
        }

        bin_size[ bid_x + mbinx * ( bid_y + bid_z * mbiny ) ]++;
    }
    int bsize_avg = 0, bsize_max = 0, bsize_min = 9999999;
    for(int i=0;i<bin_size.size();i++) {
    	bsize_avg += bin_size[i];
    	bsize_max = std::max( bsize_max, bin_size[i] );
    	bsize_min = std::min( bsize_min, bin_size[i] );
    }
    printf("my_box LO %lf %lf %lf HI %lf %lf %lf\n", my_box.lo.x, my_box.lo.y, my_box.lo.z, my_box.hi.x, my_box.hi.y, my_box.hi.z);
    printf("mbin %d %d %d\n", mbinx, mbiny, mbinz);
    printf("bin size %lf %lf %lf\n", 1/bininvx, 1/bininvy, 1/bininvz );
    printf("CPU bin max %d min %d avg %d\n", bsize_max, bsize_min, bsize_avg / bin_size.size() );
#endif

    int threads_per_block = meso_device->query_block_size( gpu_assign_bin_id );
    gpu_assign_bin_id <<< n_block( n_atom, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
    	meso_atom->dev_coord[0],
        meso_atom->dev_coord[1],
        meso_atom->dev_coord[2],
        cuda_bin.dev_bin_id,
        cuda_bin.dev_atm_id,
        my_box.lo, my_box.hi,
        mbinx, mbiny, mbinz,
        bininvx, bininvy, bininvz,
        atom->nlocal, n_atom );

    sorter.sort( cuda_bin.dev_bin_id, cuda_bin.dev_atm_id, n_atom, bin_id_width, meso_device->stream() );

//  // DEBUG: check AID
//  cuda_engine->sync_device();
//  vector<int> AID( n_atom );
//  vector<uint> BID( n_atom );
//  cudaMemcpy( &AID[0], cudaBin.devAIDList, AID.size()*sizeof(int), cudaMemcpyDefault);
//  cudaMemcpy( &BID[0], cudaBin.devBIDList, BID.size()*sizeof(uint), cudaMemcpyDefault);
//  cuda_engine->sync_device();
//  uint bid_last = -1;
//  for( int i = 0 ; i < AID.size() ; i++ )
//  {
//      cout<<AID[i]<<" "<<BID[i]<<endl;
//      bid_last = BID[i] ;
//  }
//  cudaDeviceReset();
//  exit(0);

    threads_per_block = meso_device->query_block_size( gpu_find_bin_boundary );
    gpu_find_bin_boundary <<< n_block( n_atom, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        cuda_bin.dev_bin_id,
        cuda_bin.dev_atm_id,
        cuda_bin.dev_bin_location,
        cuda_bin.dev_bin_isghost,
        n_bin,
        atom->nlocal,
        n_atom );

    threads_per_block = meso_device->query_block_size( gpu_calc_bin_size );
    gpu_calc_bin_size <<< n_block( n_bin, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        cuda_bin.dev_bin_location,
        cuda_bin.dev_bin_isghost,
        cuda_bin.dev_bin_size,
        cuda_bin.dev_bin_size_local,
        n_bin );

    *max_local_bin_size = 0;
    threads_per_block = 1024 ;
    gpu_reduce_max_host <<< 1, threads_per_block, 0, meso_device->stream() >>> (
        cuda_bin.dev_bin_size_local.ptr(),
        max_local_bin_size.ptr(),
        0,
        n_bin );

//  vector<u64> bid(n_atom);
//  vector<uint> aid(n_atom);
//  vector<int> bhead(n_bin+1), isghost(n_bin);
//  cudaMemcpyAsync( &bid[0], cudaBin.devBIDList, bid.size() * sizeof(u64), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &aid[0], cudaBin.devAIDList, aid.size() * sizeof(uint), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &bhead[0], cudaBin.devBinLoc, bhead.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &isghost[0], cudaBin.devBinIsGhost, isghost.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cuda_engine->sync_device();
//  for(int i = 0 ; i < n_bin ; i++ )
//  {
//      cout<<"bid = "<< i <<", "<< bhead[i]<<" - " << bhead[i+1] ;
//      if ( isghost[i] ) cout<<"*";
//      cout<<endl;
//      for( int j = bhead[i] ; j < bhead[i+1] ; j++ )
//      {
//          cout << '\t' << bid[j] << ' ' << aid[j] << endl;
//      }
//      cout<<"_________________________________________________"<<endl;
//  }
//  cout<<"Max Bin contain "<<MaxLocalBinSize[0]<<" particles"<<endl;
//  cudaDeviceReset();
//  exit(0);

    // expand the stencil from the LAMMPS definition to my definition
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_expand_stencil, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_expand_stencil, cudaFuncCachePreferL1 );
    }

    uint tuple_size = max( 8, ( int )pow( 2.0, ( int )log2( expected_bin_size / 4 ) ) );
    grid_cfg.x = ceiling( n_bin * tuple_size, grid_cfg.y ) / grid_cfg.y;
    gpu_expand_stencil <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        cuda_bin.dev_atm_id,
        cuda_bin.dev_bin_location,
        ghost ? cuda_bin.dev_bin_size : cuda_bin.dev_bin_size_local,
        list->dev_neighbor_count,
        list->dev_neighbor_bin,
        list->dev_stencil,
        list->dev_stencil_len,
        list->dev_neighbor_bin.pitch(),
        list->dev_stencil.pitch(),
        tuple_size,
        n_bin
    );

    {
        static GridConfig grid_cfg;
        if( !grid_cfg.x )
            grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_sentinel, 0, cudaFuncCachePreferShared );

        gpu_sentinel <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            list->dev_stencil_len,
            cuda_bin.dev_bin_size_local,
            list->dev_stencil.pitch(),
            n_bin
        );
    }


//  cuda_engine->stream().sync();
//  cout<< cudaGetErrorString( cudaGetLastError() ) <<endl;
//  vector<int> stencil_len( n_bin, 0 );
//  vector<int> local_size( n_bin, 0 );
//  vector<int> stencil( n_bin * list->stencil_padding, 0 );
//  cudaMemcpyAsync( &stencil_len[0], list->devStencilLen, stencil_len.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &local_size [0], cudaBin.devBinLocalSize, local_size.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &stencil    [0], list->devStencil   , stencil.size()     * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cuda_engine->stream().sync();
//  for( int i = 5684 ; i < 5784; i++ )
//  {
//      if ( !local_size[i] ) continue;
//      cout<<"Bin "<<i<<" : ";//<<stencil_len[i]<<endl;;
//      for( int j = 0 ; j < stencil_len[i] ; j++ ) cout<<stencil[ i * list->stencil_padding + j ]<<' ';
//      cout<<endl;
//  }
//  cudaDeviceReset();
//  cout<<"exit at "<<__LINE__<<"@"<<__FILE__<<endl;
//  exit(0);
}

__global__ void gpu_stencil_full_bin_3d(
    int *neighbor_count,
    int *neighbor_bin,
    const int  mbinx,
    const int  mbiny,
    const int  mbinz,
    const int  neighbor_range,
    const int  neighbin_padding
)
{
    int BidX = blockIdx.x * blockDim.x + threadIdx.x;
    int BidY = blockIdx.y * blockDim.y + threadIdx.y;
    int BidZ = blockIdx.z * blockDim.z + threadIdx.z;

    if( BidX >= mbinx || BidY >= mbiny || BidZ >= mbinz ) return;

    int  Bid = calc_bid( BidX, BidY, BidZ, mbinx, mbiny );
    int  nStencil = 0;
    uint* NeighBinMorton = new uint[neighbin_padding];
    neighbor_bin += Bid * neighbin_padding ;

    for( int k = -neighbor_range ; k <= neighbor_range ; k++ )
        for( int j = -neighbor_range ; j <= neighbor_range ; j++ )
            for( int i = -neighbor_range ; i <= neighbor_range ; i++ ) {
                int NewBidX = BidX + i ;
                int NewBidY = BidY + j ;
                int NewBidZ = BidZ + k ;
                if( NewBidX < 0 || NewBidX >= mbinx
                        || NewBidY < 0 || NewBidY >= mbiny
                        || NewBidZ < 0 || NewBidZ >= mbinz ) continue;
                neighbor_bin   [ nStencil ] = calc_bid( NewBidX, NewBidY, NewBidZ, mbinx, mbiny ) ;
                NeighBinMorton[ nStencil ] = morton_encode( NewBidX, NewBidY, NewBidZ );
                if( NewBidX == 0 || NewBidX == mbinx - 1
                        || NewBidY == 0 || NewBidY == mbiny - 1
                        || NewBidZ == 0 || NewBidZ == mbinz - 1 ) NeighBinMorton[ nStencil ] += 0x80000000;
                nStencil++;
            }

    // sort: bubble sort!
    for( int i = 0 ; i < nStencil ; i++ )
        for( int j = i + 1 ; j < nStencil ; j++ ) {
            if( NeighBinMorton[ i ] > NeighBinMorton[ j ] ) {
                int tmp1 = neighbor_bin[ i ];
                neighbor_bin[ i ] = neighbor_bin[ j ];
                neighbor_bin[ j ] = tmp1 ;
                uint tmp2 = NeighBinMorton[ i ];
                NeighBinMorton[ i ] = NeighBinMorton[ j ];
                NeighBinMorton[ j ] = tmp2 ;
            }
        }

    neighbor_count[ Bid ] = nStencil;
    delete [] NeighBinMorton;
}

void MesoNeighbor::stencil_full_bin_3d_meso( NeighList *list, int sx, int sy, int sz )
{
    MesoNeighList *dlist = lists_device[ list->index ];
    if( dlist == NULL ) error->all( FLERR, "<MESO> neighbor list not cudable" );

    dlist->stencil_allocate( mbinx * mbiny * mbinz, 0 );

    int binsize_optimal = neighbor->cutneighmax;
    int neighbor_range = floor( cutneighmax / binsize_optimal + 0.5 );
    dim3 grid_size, block_size;
    block_size.x = block_size.y = block_size.z = 8;
    grid_size.x = n_block( mbinx, block_size.x );
    grid_size.y = n_block( mbiny, block_size.y );
    grid_size.z = n_block( mbinz, block_size.z );

    gpu_stencil_full_bin_3d <<< grid_size, block_size, 0, meso_device->stream() >>> (
        dlist->dev_neighbor_count,
        dlist->dev_neighbor_bin,
        mbinx,
        mbiny,
        mbinz,
        neighbor_range,
        dlist->dev_neighbor_bin.pitch()
    );

//  vector<int> stencil( list->neighbin_padding * list->nBinMax );
//  vector<int> stencilcount( list->nBinMax );
//  cudaMemcpyAsync( &stencil[0], list->devNeighborBin, stencil.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cudaMemcpyAsync( &stencilcount[0], list->devNeighborCount, stencilcount.size() * sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//  cuda_engine->stream().sync();
//  for(int i=0;i<mbinx*mbiny*mbinz;i++)
//  {
//      cout<<"bin "<<i<<" has "<<stencilcount[i]<<" stencils."<<endl;
//      for(int j=0;j<stencilcount[i];j++)
//      {
//          cout << stencil[ i * list->neighbin_padding + j ] <<',' << flush;
//      }
//      cout<<endl;
//  }
//  cudaDeviceReset();
//  exit(0);
}

void MesoNeighbor::setup_bins()
{
    // GlobalBox entire domain
    // HaloBox   my subdomain extended by comm->cutghost
    // MyBox     my subdomain
    double3 GlobalDim, MyDim; // HaloDim;
    double3 GhostCutoff = make_double3( comm->cutghost[0], comm->cutghost[1], comm->cutghost[2] );
    BoxDim GlobalBox( make_double3( domain->boxhi[0], domain->boxhi[1], domain->boxhi[2] ),
                      make_double3( domain->boxlo[0], domain->boxlo[1], domain->boxlo[2] ) );
    BoxDim MyBox( make_double3( domain->subhi[0], domain->subhi[1], domain->subhi[2] ),
                  make_double3( domain->sublo[0], domain->sublo[1], domain->sublo[2] ) );
    BoxDim HaloBox( MyBox.hi + GhostCutoff, MyBox.lo - GhostCutoff );

    //determine each domain size
    GlobalDim = GlobalBox.hi - GlobalBox.lo ;
    MyDim     = MyBox    .hi - MyBox    .lo ;
//  HaloDim   = HaloBox  .hi - HaloBox  .lo ;

    // calculate particle density in sub box
    double MyBoxVolume   = MyDim.x * MyDim.y * MyDim.z ;
//  double HaloBoxVolume = HaloDim.x * HaloDim.y + HaloDim.z ;

    this->my_box = MyBox ;
    this->local_particle_density = atom->nlocal / MyBoxVolume;
    if ( this->local_particle_density < 3 ) this->local_particle_density = 3;
    this->expected_neigh_count = local_particle_density * ( 4.0 / 3.0 * 3.142 * pow( cutneighmax, 3.0 ) ) ;
    this->expected_neigh_count *= 4.0 ;
    this->expected_neigh_count = max( expected_neigh_count, ( double )max( 32, atom->maxspecial ) );
#ifdef LMP_MESO_LOG_L2
    fprintf( stderr, "<MESO> Expected neighbor count (max pair num) of each particle: %.1lf\n", expected_neigh_count );
#endif

    double binsize_optimal = 1.0 * cutneighmax ;
    this->expected_bin_size = local_particle_density * binsize_optimal * binsize_optimal * binsize_optimal ;
#ifdef LMP_MESO_LOG_L2
    fprintf( stderr, "<MESO> Expected bin size: %.1lf\n", expected_bin_size );
#endif

    // test for too many global bins in any dimension due to huge global domain
    r64 BinSizeInv = 1.0 / binsize_optimal;
    if( GlobalDim.x * BinSizeInv > INT_MAX
            || GlobalDim.y * BinSizeInv > INT_MAX
            || GlobalDim.z * BinSizeInv > INT_MAX )
        error->all( FLERR, "Domain too large for neighbor bins" );

    // create actual bins
    nbinx = max( ( int )( GlobalDim.x * BinSizeInv ), 1 );
    nbiny = max( ( int )( GlobalDim.y * BinSizeInv ), 1 );
    nbinz = max( ( int )( GlobalDim.z * BinSizeInv ), 1 );
    // bin boundary aligned with local box, +1 bin on each side for ghost particles
    mbinx = max( ( int )( MyDim.x * BinSizeInv ), 1 ) + 2;
    mbiny = max( ( int )( MyDim.y * BinSizeInv ), 1 ) + 2;
    mbinz = max( ( int )( MyDim.z * BinSizeInv ), 1 ) + 2;
    binsizex = MyDim.x / ( mbinx - 2 );
    binsizey = MyDim.y / ( mbiny - 2 );
    binsizez = MyDim.z / ( mbinz - 2 );
    bininvx = 1.0 / binsizex;
    bininvy = 1.0 / binsizey;
    bininvz = 1.0 / binsizez;

    if( binsize_optimal * bininvx > CUT2BIN_RATIO ||
            binsize_optimal * bininvy > CUT2BIN_RATIO ||
            binsize_optimal * bininvz > CUT2BIN_RATIO ) error->all( FLERR, "Cannot use neighbor bins - box size << cutoff" );

#ifdef LMP_MESO_LOG_L2
    fprintf( stderr, "<MESO> Bin grid for current domain: %d x %d x %d\n", mbinx, mbiny, mbinz );
#endif

    // create stencils for pairwise neighbor lists
    // only done for lists with stencilflag and buildflag set
    for( int i = 0; i < nslist; i++ ) {
        stencil_full_bin_3d_meso( lists[ slist[i] ], sx, sy, sz );
    }
}

