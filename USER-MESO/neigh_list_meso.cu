#include "memory.h"
#include "neigh_request.h"
#include "error.h"

#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;

MesoNeighList::MesoNeighList( LAMMPS *lmp ) : Pointers( lmp ), MesoPointers( lmp ),
    dev_neighbor_count( lmp, "MesoNeighList::dev_neighbor_count" ),
    dev_neighbor_bin( lmp, "MesoNeighList::dev_neighbor_bin" ),
    dev_stencil( lmp, "MesoNeighList::dev_stencil" ),
    dev_stencil_len( lmp, "MesoNeighList::dev_stencil_len" ),
    dev_pair_count_core( lmp, "MesoNeighList::dev_pair_count_core" ),
    dev_pair_count_skin( lmp, "MesoNeighList::dev_pair_count_skin" ),
    dev_pair_table( lmp, "MesoNeighList::dev_pair_table" ),
    dev_tail_count( lmp, "MesoNeighList::dev_tail_count" ),
    dev_tail( lmp, "MesoNeighList::dev_tail" )
{
    n_max            = 0;
    n_bin_max        = 0;
    n_row = 0;
    n_col = 0;
}

MesoNeighList::~MesoNeighList()
{
}

void MesoNeighList::grow( int nmax )
{
    if( nmax <= n_max ) return;

    n_max = nmax ;
    n_row = ceiling( n_max, 32 );
    n_col = ceiling( meso_neighbor->expected_neigh_count, 32 );
    dev_pair_count_core.grow( n_max, false, false );
    dev_pair_count_skin.grow( n_max, false, false );
    dev_pair_table.grow( n_row * n_col, false, false );
//  devTail.grow( PairPaddingRow * PairPaddingCol * 0.125, false, false );
//  devTailCount.grow( 1, false, false );
}

uint MesoNeighList::grow_prune_buffer( int each_buffer_len, int PruneBufferCount )
{
    uint padding = 0;
//  uint padding = ceiling( each_buffer_len, 32 );
//  if ( devPruneBuffer == NULL || cuda_engine->QueryMemSize( devPruneBuffer ) < padding * PruneBufferCount * sizeof(int2) )
//  {
//      cout<<"<MESO> grow prune buffer: " << padding << " * " << PruneBufferCount << std::endl;
//      cuda_engine->ReallocDevice( devPruneBuffer, padding * PruneBufferCount, false, false );
//  }
    return padding;
}

void MesoNeighList::stencil_allocate( int smax, int style )
{
    if( smax <= n_bin_max ) return;

    n_bin_max = smax ;
    int binsize_optimal = neighbor->cutneighmax;
    int most_bins       = pow( ceil( 2.0 * meso_neighbor->cutneighmax / binsize_optimal + 1.0 ), 3 );
    int bins_padding    = ceiling( most_bins, 32 );
    int stencil_padding = max( 1.0, most_bins * meso_neighbor->expected_bin_size * 2.25 );
    stencil_padding     = ceiling( stencil_padding, 32 );
#ifdef LMP_MESO_LOG_L2
    fprintf( stderr, "<MESO> bins_padding %d\n", bins_padding );
    fprintf( stderr, "<MESO> stencil_padding %d\n", stencil_padding );
#endif

    dev_neighbor_count.grow( n_bin_max, false );
    dev_neighbor_bin  .grow( bins_padding, n_bin_max, false );
    dev_stencil_len   .grow( n_bin_max, false );
    dev_stencil       .grow( stencil_padding, n_bin_max, false );

    if( atom->nlocal == 0 ) fprintf( stderr, "[CDEV] atom->nlocal is 0 when allocating common neighbor mask.\n" );
}

// dump the pair table after transposition
void MesoNeighList::dump_core_post_transposition( const char file[], int beg, int end )
{
    std::ofstream fout( file, std::ios_base::binary );

    std::vector<int> n_core( n_max );
    std::vector<int> pair_table( n_col * n_row, 0 );
    meso_device->sync_device();
    dev_pair_count_core.download( &n_core[0], n_core.size() );
    dev_pair_table.download( &pair_table[0], pair_table.size() );
    meso_device->sync_device();

    for( int i = beg ; i < end ; i++ ) {
        if( i >= atom->nlocal ) fout << "*\t";
        for( int j = 0 ; j < n_core[i] ; j++ ) {
            int pLocal = i % 32;
            int pBlock = i - pLocal;
            int pLine = j % 32;
            int pSOA = j / 32;
            fout << pair_table[( pBlock + pLine ) * n_col + pSOA * 32 + pLocal ] << '\t';
        }
        fout << std::endl;
    }
}

void MesoNeighList::generate_interaction_map( const char file[] )
{
    std::ofstream fout( file );

    std::vector<int> n_core( n_max );
    std::vector<int> pair_table( n_col * n_row, 0 );
    std::vector<int> tag( atom->nlocal + atom->nghost );
    meso_device->sync_device();
    ( *meso_atom->dev_tag ).download( &tag[0], tag.size() );
    dev_pair_count_core.download( &n_core[0], n_core.size() );
    dev_pair_table.download( &pair_table[0], pair_table.size() );
    meso_device->sync_device();

    int L = atom->nlocal + 1;
    std::vector<int> map( L * L, 0 );

    // DUMP THE CORE TABLE
    for( int i = 0 ; i < atom->nlocal ; i++ ) {
        for( int j = 0 ; j < n_core[i] ; j++ ) {
            int x = tag[i];
            int y = tag[ pair_table[ i * n_col + j ] ];
            if( x < y ) map[ x * L + y ]++;
            else map[ y * L + x ]++;
        }
    }
    for( int i = 0; i < L; i++ ) {
        for( int j = 0; j < L; j++ )
            fout << map[i * L + j] << ( ( j < L - 1 ) ? '\t' : '\n' );
    }
    fout.close();
}
