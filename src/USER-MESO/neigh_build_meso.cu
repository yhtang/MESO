#include "atom_vec.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "neigh_list.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "bin_meso.h"
#include "comm_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;

// persistent blocks
// using slots, I can bound the shared memory usage to less than 48KB
// for 64 active warps (64 bins)
template<int SLOT_PER_WARP, int NWORD_PER_SLOT>
__global__ void __launch_bounds__( 128, 16 ) gpu_build_neighbor_list(
    texobj tex_aid,
    texobj tex_coord_merged,
    int* __restrict pair_count_core,
    int* __restrict pair_count_skin,
    int* __restrict pair_table,
    const int* __restrict bin_head,
    const int* __restrict bin_size,
    const int* __restrict stencil_len,
    const int* __restrict stencil,
    const int  stencil_padding,
    const int  pair_buffer_padding,
    const r32  rc2_core,
    const r32  rc2_tail,
    const int  nbin,
    const int  n_part
)
{
    const int warp_per_part = __warp_num_global() / n_part;
    const int part_id = __warpid_global() / warp_per_part;
    if( part_id >= n_part ) return;
    const int id_in_partition = __warpid_global() % warp_per_part;

    if( SLOT_PER_WARP > WARPSZ )  // no register overhead because can be eliminated entirely if predicate is static
        printf( "<MESO> slot number %d exceeds warp size %d\n", SLOT_PER_WARP, WARPSZ );
    extern __shared__ int SMEM[];

    const int  lane_id = __laneid();
    const bool leader  = ( lane_id == 0 );

    int   *aid   = ( int* )   &SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 0 * SLOT_PER_WARP ];
    short *n_core = ( short* )&SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 1 * SLOT_PER_WARP ];
    short *n_skin = ( short* )&SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 1 * SLOT_PER_WARP + SLOT_PER_WARP / 2 ];
    r32   *x     = ( r32* )   &SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 2 * SLOT_PER_WARP ];
    r32   *y     = ( r32* )   &SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 3 * SLOT_PER_WARP ];
    r32   *z     = ( r32* )   &SMEM[ __warpid_local() * SLOT_PER_WARP * NWORD_PER_SLOT + 4 * SLOT_PER_WARP ];

    for( int bin_id = id_in_partition; bin_id < nbin; bin_id += warp_per_part ) {
        int cur_bin_size  = bin_size[ bin_id ] ;
        if( !cur_bin_size ) continue;

        for( int p = 0 ; p < cur_bin_size ; p += SLOT_PER_WARP * n_part ) {
            // load shared data
            int pack = min( ( cur_bin_size - p + n_part - 1 - part_id ) / n_part, SLOT_PER_WARP );
            if( lane_id < pack ) {
                aid[lane_id] = tex1Dfetch<int> ( tex_aid, bin_head[ bin_id ] + p + lane_id * n_part + part_id );
                float4 v     = tex1Dfetch<float4>( tex_coord_merged, aid[lane_id] );
                n_core[ lane_id ] = 0;
                n_skin[ lane_id ] = 0;
                x[lane_id] = v.x;
                y[lane_id] = v.y;
                z[lane_id] = v.z;
            }

            // load batch of atoms from stencil and compare against center atom
            for( int j = lane_id ; j < stencil_len[ bin_id ] ; j += warpSize ) {
                int aid_j = stencil[ bin_id * stencil_padding + j ];
                r32  xj, yj, zj;
                {
                    float4 v = tex1Dfetch<float4>( tex_coord_merged, aid_j );
                    xj = v.x, yj = v.y, zj = v.z;
                }

                // examine stencil atoms against each atom in my bin
                for( int i = 0 ; i < pack ; i++ ) {
                    r32 dx    = x[i] - xj ;
                    r32 dy    = y[i] - yj ;
                    r32 dz    = z[i] - zj ;
                    r32 dr2   = dx * dx + dy * dy + dz * dz ;

                    bool p1 = ( dr2 <= rc2_core );
                    bool p2 = ( dr2 <= rc2_tail );
                    bool p3 = ( aid[i] != aid_j );

                    int  is_hit, my_insert;
                    uint n_hit;
                    // CORE PART
                    is_hit = p1 && p3;
                    n_hit  = __ballot( is_hit );
                    my_insert = n_core[ i ] + __popc( n_hit & __lanemask_lt() );
                    if( is_hit ) pair_table[ aid[i] * pair_buffer_padding + my_insert ] = aid_j;
                    if( leader ) n_core[ i ] += __popc( n_hit );

                    // SKIN PART
                    is_hit = !p1 && p2 && p3;
                    n_hit  = __ballot( is_hit );
                    my_insert = n_skin[ i ] + __popc( n_hit & __lanemask_lt() );
                    if( is_hit ) pair_table[( aid[i] + 1 ) * pair_buffer_padding - 1 - my_insert ] = aid_j;
                    if( leader ) n_skin[ i ] += __popc( n_hit );
                }
            }

            if( lane_id < pack ) {
                pair_count_core[ aid[lane_id] ] = n_core[lane_id];
                pair_count_skin[ aid[lane_id] ] = n_skin[lane_id];
            }
        }
    }
}

template<int SQ_SIZE>
__global__ void gpu_prune_neigh_list(
    int*  __restrict pair_count_core,
    int*  __restrict pair_count_skin,
    int*  __restrict pair_table,
    int*  __restrict TailCount,
    int2* __restrict Tail,
    const int pair_buffer_padding,
    const int n_atom
)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int subLaneID = __laneid() % SQ_SIZE;
    int n_core = 0;
    int n_skin = 0;
    int pIns  = 0;

    // only prune the skin part
    if( gid < n_atom ) {
        n_core = pair_count_core[gid];
        n_skin = pair_count_skin[gid];
    }
    int np      = n_core + n_skin;
    int nPairAvg   = __warp_sum( np ) / __popc( __ballot( 1 ) );
    int nPrune     = max( min( np - nPairAvg, n_skin ), 0 ); // max(0): cannot increase pair count.., min(skin): only cut skin pairs
    int nPruneCumu = __warp_prefix_excl( nPrune );

    if( __laneid() == WARPSZ - 1 ) pIns = atomic_add( TailCount, nPruneCumu + nPrune );
    pIns = __shfl( pIns, WARPSZ - 1 );

    for( int i = 0 ; i < SQ_SIZE ; i++ ) {
        int nCut   = __shfl( nPrune    , i, SQ_SIZE );
        int nTotal = __shfl( n_skin    , i, SQ_SIZE );
        int GID    = __shfl( gid       , i, SQ_SIZE );
        int pInc   = __shfl( nPruneCumu, i, SQ_SIZE );
        for( int p = subLaneID; p < nCut; p += SQ_SIZE ) {
//          Tail[ pIns + pInc + p ] = make_int2( GID, pair_table[ GID * pair_buffer_padding + np - nCut + p ] );
            Tail[ pIns + pInc + p ] = make_int2( GID, pair_table[( GID + 1 ) * pair_buffer_padding - 1 - ( nTotal - nCut + p ) ] );
        }
    }

    if( gid < n_atom ) pair_count_skin[gid] = n_skin - nPrune;
}

// join core & skin
__global__ void gpu_join_neigh_list(
    int* __restrict pair_count_core,
    int* __restrict pair_count_skin,
    int* __restrict pair_table,
    const int pair_buffer_padding,
    const int n_atom
)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_core = 0;
    int n_skin = 0;

    if( gid < n_atom ) {
        n_core = pair_count_core[gid];
        n_skin = pair_count_skin[gid];
        pair_count_core[gid] = n_core + n_skin;
    }

    int lane_id = __laneid();
    int base   = __shfl( gid, 0 ) * pair_buffer_padding;

#pragma unroll
    for( int i = 0 ; i < WARPSZ; i++ ) {
        int n1 = __shfl( n_core, i );
        int n2 = __shfl( n_skin, i );
        int start = n1 - ( n1 % WARPSZ );
        int front = base + n1;
        int rear  = pair_buffer_padding - n2;
        for( int p = start + lane_id ; p < pair_buffer_padding; p += WARPSZ ) {
            if( p >= rear )
                pair_table[ front + ( p - rear ) ] = pair_table[ base + p ];
        }
        base += pair_buffer_padding;
    }
}

// 32x8
__global__ void gpu_transpose_neigh_list(
    int* __restrict n_pair,
    int* __restrict pair_table,
    const int pair_buffer_padding,
    const int n_atom )
{
    __shared__ int nMax;
    __shared__ int np[32];
    __shared__ int Buffer[32][33];

    if( threadIdx.y == 0 ) {
        int gid = blockIdx.x * 32 + threadIdx.x;
        np[threadIdx.x] = ( gid < n_atom ) ? ( n_pair[ gid ] ) : ( 0 );
        int nMaxPair = __warp_max( np[threadIdx.x] );
        if( threadIdx.x == 0 ) nMax = nMaxPair;
    }
    __syncthreads();

    int xIndex = threadIdx.x;
    int yIndex = threadIdx.y + blockIdx.x * 32;
    int index = xIndex + yIndex * pair_buffer_padding;

    for( int p = 0 ; p < nMax ; p += 32 ) {
#pragma unroll
        for( int i = 0 ; i < 32; i += 8 ) {
//          if ( p + threadIdx.x < np[ threadIdx.y + i ] )
            Buffer[ threadIdx.y + i ][ threadIdx.x ] = pair_table[ index + i * pair_buffer_padding + p ];
        }
        __syncthreads();

#pragma unroll
        for( int i = 0 ; i < 32; i += 8 ) {
//          if ( p + threadIdx.y + i < np[ threadIdx.x ] )
            pair_table[ index + i * pair_buffer_padding + p ] = Buffer[ threadIdx.x ][ threadIdx.y + i ];
        }
        __syncthreads();
    }
}

__global__ void gpu_sentinel_nb(
    int* __restrict pair_count_core,
    int* __restrict pair_count_skin,
    const int pair_buffer_padding,
    const int nAll )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x ; i < nAll ; i += gridDim.x * blockDim.x ) {
        if( pair_count_core[i] + pair_count_skin[i] > pair_buffer_padding )
            printf( "<MESO> Pair table overflow: %d + %d > %d; local density too high for particle %d\n", pair_count_core[i], pair_count_skin[i], pair_buffer_padding, i );
    }
}

void MesoNeighbor::full_bin_meso( NeighList *list )
{
    const static int nslot = 32;
    const static int nword = 5;

    // bin local & ghost atoms
    MesoNeighList *dlist = lists_device[ list->index ];
    if( dlist == NULL ) error->all( FLERR, "<MESO> neighbor list not cudable" );

    binning_meso( dlist, false );
    dlist->grow( atom->nlocal );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, atom->nlocal + atom->nghost, true );

    static GridConfig grid_cfg, warp_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.left_peak( 0, gpu_build_neighbor_list<nslot, nword>, 0, cudaFuncCachePreferShared );
        warp_cfg = make_int2( grid_cfg.y / WARPSZ, 1 );
        cudaFuncSetCacheConfig( gpu_build_neighbor_list<nslot, nword>, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_prune_neigh_list<8>, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_join_neigh_list, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_transpose_neigh_list, cudaFuncCachePreferShared );
    }

    int nbin       = mbinx * mbiny * mbinz;
    int shmem_size = ( grid_cfg.y / WARPSZ ) * ( nslot * nword * sizeof( int ) );

	gpu_build_neighbor_list<nslot, nword> <<< grid_cfg.x, grid_cfg.y, shmem_size, meso_device->stream() >>> (
		cuda_bin.tex_atm_id,
		meso_atom->tex_coord_merged,
		dlist->dev_pair_count_core,
		dlist->dev_pair_count_skin,
		dlist->dev_pair_table,
		cuda_bin.dev_bin_location,
		cuda_bin.dev_bin_size_local,
		dlist->dev_stencil_len,
		dlist->dev_stencil,
		dlist->dev_stencil.pitch(),
		dlist->n_col,
		pow( cutneighmax - skin, 2.0 ),
		pow( cutneighmax, 2.0 ),
		nbin,
		warp_cfg.partition( nbin, meso_neighbor->expected_bin_size, 0.0 ) );

    static GridConfig grid_cfg2;
    if( !grid_cfg2.x )
        grid_cfg2 = meso_device->occu_calc.right_peak( 0, gpu_sentinel_nb, 0, cudaFuncCachePreferShared );
    gpu_sentinel_nb <<< grid_cfg2.x, grid_cfg2.y, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_count_skin,
        dlist->n_col,
        atom->nlocal );

//  dlist->devTailCount.set( 0, cuda_engine->stream() );

    int threads_per_block;

//  threads_per_block = cuda_engine->query_block_size( (void*)gpu_prune_neigh_list<8> );
//  gpu_prune_neigh_list<8><<<n_block(atom->nlocal,threads_per_block),threads_per_block,0,cuda_engine->stream()>>> (
//      dlist->devPairCountCore,
//      dlist->devPairCountSkin,
//      dlist->devPairTable,
//      dlist->devTailCount,
//      dlist->devTail,
//      dlist->PairPaddingCol,
//      atom->nlocal );

    threads_per_block = meso_device->query_block_size( gpu_join_neigh_list );
    gpu_join_neigh_list <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_count_skin,
        dlist->dev_pair_table,
        dlist->n_col,
        atom->nlocal );

    if( atom->special ) filter_exclusion_meso( dlist );

    //  dump the pair table before transposition
//  if ( update->ntimestep == 300 )
//  {
//    using namespace std;
//      vector<int> n_core( atom->nlocal );
//      vector<int> n_skin( atom->nlocal );
//      vector<int> pair_table( dlist->n_col * dlist->n_row, 0);
//      vector<int> tag( atom->nlocal+atom->nghost );
//
//      dlist->dev_pair_count_core.download( &n_core[0], n_core.size() );
//      dlist->dev_pair_count_skin.download( &n_skin[0], n_skin.size() );
//      dlist->dev_pair_table.download( &pair_table[0], pair_table.size() );
//      (*meso_atom->dev_tag).download( &tag[0], tag.size() );
//      meso_device->sync_device();
//
//
//      vector<int> tag2idx( atom->nlocal, 0 );
//      for( int i = 0 ; i < atom->nlocal ; i++ ) tag2idx[ tag[i] - 1 ] = i;
//
//      ofstream fout("pairtable.gpu.original", ios_base::binary);
//      // DUMP THE CORE TABLE
//      int sum = 0;
//      vector<int> ind;
//      for(int t = 0 ; t < atom->nlocal ; t++ )
//      {
//          int i = tag2idx[t];
//          fout<<"tag\t"<<i<<"\tindex\t"<<t<<"\tn\t"<<n_core[i]<<":\t";
//          if ( n_core[i] >= dlist->n_col )
//          {
//              cout<<i<<' '<<n_core[i]<<">="<<dlist->n_col<<endl;
//          }
//          sum += n_core[i];
//          // dump tag of neighbor
//          ind.clear();
//          for( int j = 0 ; j < n_core[i] ; j++ )
//              ind.push_back( tag[ pair_table[ i * dlist->n_col + j ] ] );
//          sort( ind.begin(), ind.end() );
//          for( int p = 0 ; p < ind.size() ; p++ ) fout<< ind[p] <<'\t';
//          fout<<endl;
//      }
//      cout<<"sum = "<<sum<<endl;
//      fast_exit(0);
//  }
    //dlist->generate_interaction_map( "interaction.gpu" );

    dim3 threadsConfig( 32, 8, 1 );
    gpu_transpose_neigh_list <<< n_block( atom->nlocal, 32 ), threadsConfig, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_table,
        dlist->n_col,
        atom->nlocal );

//  if ( update->ntimestep > 2000 )
//  {
//      unsigned long long sum = 0;
//      vector<int> NPairList( atom->nlocal );
//      vector<int> pair_table( dlist->pairtable_padding * ExpectedNeighCount, 0);
//      cudaMemcpyAsync( &NPairList[0], dlist->devPairCountList, NPairList.size()*sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//      cudaMemcpyAsync( &pair_table[0], dlist->devPairTable,     pair_table.size()*sizeof(int), cudaMemcpyDefault, cuda_engine->stream() );
//      cuda_engine->stream().sync();
//      cout<< cudaGetErrorString( cudaGetLastError() )<<endl;
//
//      ofstream fout("pairtable.gpu.original", ios_base::binary);
//      for(int i = 0 ; i < atom->nlocal ; i++ )
//      {
//          sum += NPairList[i];
//          if ( NPairList[i] >= ExpectedNeighCount )
//          {
//              cout<<i<<' '<<NPairList[i]<<">="<<ExpectedNeighCount<<endl;
//              char d;
//              cin.get(d);
//          }
//          for(int j=0;j<NPairList[i];j++)
//          {
//              if ( pair_table[ i + j * dlist->pairtable_padding ] != 0x7FFFFFFF )
//                  fout<< pair_table[ i + j * dlist->pairtable_padding ] <<' ';
//          }
//          fout<<endl;
//      }
//      cout<<"average pair count: "<< (double)sum / atom->nlocal <<endl;
//      cout<<"[CDEV] pair table dumped to pairtable.gpu.index at "<<__LINE__<<"@"<<__FILE__<<endl;
//      cudaDeviceReset();
//      exit(0);
//  }

    //dlist->dump_core_post_transposition( "pairtable.gpu.transpose", 0, atom->nlocal );
}

void MesoNeighbor::full_bin_meso_ghost( NeighList *list )
{
    const static int nslot = 32;
    const static int nword = 5;

    // bin local & ghost atoms
    MesoNeighList *dlist = lists_device[ list->index ];
    if( dlist == NULL ) error->all( FLERR, "<MESO> neighbor list not cudable" );

    int n_particle = atom->nlocal + atom->nghost;

    binning_meso( dlist, true );
    dlist->grow( n_particle );

    meso_atom->meso_avec->dp2sp_merged( 0, 0, n_particle, true );

    static GridConfig grid_cfg, warp_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.left_peak( 0, gpu_build_neighbor_list<nslot, nword>, 0, cudaFuncCachePreferShared );
        warp_cfg = make_int2( grid_cfg.y / WARPSZ, 1 );
        cudaFuncSetCacheConfig( gpu_build_neighbor_list<nslot, nword>, cudaFuncCachePreferShared );
        cudaFuncSetCacheConfig( gpu_prune_neigh_list<8>, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_join_neigh_list, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_transpose_neigh_list, cudaFuncCachePreferShared );
    }

    int nbin       = mbinx * mbiny * mbinz;
    int shmem_size = ( grid_cfg.y / WARPSZ ) * ( nslot * nword * sizeof( int ) );

    gpu_build_neighbor_list<nslot, nword> <<< grid_cfg.x, grid_cfg.y, shmem_size, meso_device->stream() >>> (
        cuda_bin.tex_atm_id,
        meso_atom->tex_coord_merged,
        dlist->dev_pair_count_core,
        dlist->dev_pair_count_skin,
        dlist->dev_pair_table,
        cuda_bin.dev_bin_location,
        cuda_bin.dev_bin_size,
        dlist->dev_stencil_len,
        dlist->dev_stencil,
        dlist->dev_stencil.pitch(),
        dlist->n_col,
        pow( cutneighmax - skin, 2.0 ),
        pow( cutneighmax, 2.0 ),
        nbin,
        warp_cfg.partition( nbin, meso_neighbor->expected_bin_size, 0.0 ) );

    static GridConfig grid_cfg2;
    if( !grid_cfg2.x )
        grid_cfg2 = meso_device->occu_calc.right_peak( 0, gpu_sentinel_nb, 0, cudaFuncCachePreferShared );
    gpu_sentinel_nb <<< grid_cfg2.x, grid_cfg2.y, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_count_skin,
        dlist->n_col,
        n_particle );

    int threads_per_block;

    threads_per_block = meso_device->query_block_size( gpu_join_neigh_list );
    gpu_join_neigh_list <<< n_block( n_particle, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_count_skin,
        dlist->dev_pair_table,
        dlist->n_col,
        n_particle );

    if( atom->special ) filter_exclusion_meso( dlist );

    dim3 threadsConfig( 32, 8, 1 );
    gpu_transpose_neigh_list <<< n_block( n_particle, 32 ), threadsConfig, 0, meso_device->stream() >>> (
        dlist->dev_pair_count_core,
        dlist->dev_pair_table,
        dlist->n_col,
        n_particle );

    //dlist->dump_core_post_transposition( "pairtable.gpu.transpose", 0, n_particle );
}

// each warp deal with one i particle
__global__ void gpu_filter_exclusion(
    texobj       tex_tag,
    int         *n_pair,
    int volatile *pair_table,
    int4        *n_excl,
    int         *excl_table,
    const int    pairtable_padding,
    const int    excltable_padding,
    const int    n_local,
    const int    excluded
)
{
    int lane_id = __laneid();
    for( int i = __warpid_global(); i < n_local; i += __warp_num_global() ) {
        int ne = ( excluded == 0 ? n_excl[i].x : ( excluded == 1 ? n_excl[i].y : n_excl[i].z ) );
        int np = n_pair[i];

        if( ne ) {
            int base = 0;
            for( int p = 0; p < np; p += WARPSZ ) {
                int  t, j;
                bool keep = false;
                int  M = min( WARPSZ, np - p );
                if ( lane_id < M ) {
                    keep = true;
                    j = pair_table[ i * pairtable_padding + p + lane_id ];
                    t = tex1Dfetch<int>( tex_tag, j );
                }
                for( int P = 0 ; P < ne ; P += WARPSZ ) {
					int N = min( WARPSZ, ne - P );
					int s;
					if( lane_id < N ) s = excl_table[ i * excltable_padding + P + lane_id ];

                    for( int k = 0 ; k < N ; k++ ) {
                        int o = __shfl( s, k );
                        keep = ( keep && ( t != o ) );
                    }
                }
                int n_keep = __ballot( keep );
                if( keep ) {
                    pair_table[ i * pairtable_padding + base + __popc( n_keep << ( WARPSZ - lane_id ) ) ] = j;
                }
                base += __popc( n_keep << ( WARPSZ - M ) );
            }
            if( lane_id == 0 ) n_pair[i] = base;
        }
    }
}

void MesoNeighbor::filter_exclusion_meso( MesoNeighList *dlist )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.left_peak( 0, gpu_filter_exclusion, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_filter_exclusion, cudaFuncCachePreferL1 );
    }

    int excluded = -1;
    for( int i = 1 ; i < 4 && !( force->special_lj[i] || force->special_coul[i] ); i++ ) excluded = i - 1;

    if( excluded >= 0 )
        gpu_filter_exclusion <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->tex_tag,
            dlist->dev_pair_count_core,
            dlist->dev_pair_table,
            meso_atom->dev_nexcl,
            meso_atom->dev_excl_table,
            dlist->n_col,
            meso_atom->meso_avec->excl_table_padding,
            atom->nlocal,
            excluded
        );
}


