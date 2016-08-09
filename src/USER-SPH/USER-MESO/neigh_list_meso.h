#ifndef LMP_MESO_NEIGH_LIST
#define LMP_MESO_NEIGH_LIST

#include "meso.h"

namespace LAMMPS_NS
{

class MesoNeighList : protected Pointers, protected MesoPointers
{
public:
    int  index;
    int  n_max;
    int  n_bin_max;

    DeviceScalar    <int> dev_neighbor_count;
    DevicePitched<int> dev_neighbor_bin;  // one row for each bin, store indices of bins in its neighbor in the form of [start_bin_id,end_bin_id),...
    DeviceScalar    <int> dev_stencil_len;
    DevicePitched<int> dev_stencil;      // one row for each bin, store indices of atoms in all its neighboring bins

    int  n_row;
    int  n_col;
    DeviceScalar<int>  dev_pair_count_core;
    DeviceScalar<int>  dev_pair_count_skin;
    DeviceScalar<int>  dev_pair_table; // column table
    DeviceScalar<int>  dev_tail_count; // 0: core, 1: skin
    DeviceScalar<int2> dev_tail;

    MesoNeighList( class LAMMPS * );
    ~MesoNeighList();

    void grow( int );                  // grow maxlocal
    void stencil_allocate( int, int ); // allocate stencil arrays

    uint grow_prune_buffer( int EachBufferLen, int PairBufferCount );
    void grow_common_excl();

    void dump_core_post_transposition( const char file[], int beg, int end );
    void generate_interaction_map( const char file[] );
};

}

#endif
