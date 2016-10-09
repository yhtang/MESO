#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "force.h"
#include "modify.h"
#include "fix.h"
#include "atom.h"
#include "output.h"
#include "thermo.h"
#include "update.h"
#include "domain.h"
#include "group.h"
#include "memory.h"
#include "neighbor.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"

using namespace LAMMPS_NS;

MesoAtom::MesoAtom( class LAMMPS *lmp ) : Atom( lmp ), MesoPointers( lmp ),
    sorter( lmp ),
    dev_n_bulk( lmp, "MesoAtom::dev_n_bulk" ),
    dev_map_array( lmp, "MesoAtom::dev_map_array" ),
    dev_hash_key( lmp, "MesoAtom::dev_hash_key" ),
    dev_hash_val( lmp, "MesoAtom::dev_hash_val" ),
    // textures
    tex_map_array( lmp, "MesoAtom::tex_map_array" ),
    tex_hash_key( lmp, "MesoAtom::tex_hash_key" ),
    tex_hash_val( lmp, "MesoAtom::tex_hash_val" ),
    tex_tag( lmp, "MesoAtom::tex_tag" ),
    tex_mass( lmp, "MesoAtom::tex_mass" ),
    tex_rho( lmp, "MesoAtom::tex_rho" ),
    tex_coord_merged( lmp, "MesoAtom::tex_coord_merged" ),
    tex_veloc_merged( lmp, "MesoAtom::tex_veloc_merged" ),
    hash_load_factor( 0.5 ),
    nonce( 0x4D6C3671U )
{
    bulk_counting = 0;
    n_bulk = 0;
    n_border = 0;
    hash_table_size  = 0;
    meso_avec        = NULL;
    dev_n_bulk.grow( 1 );
}

MesoAtom::~MesoAtom()
{
    tex_map_array   .unbind();
    tex_hash_key    .unbind();
    tex_hash_val    .unbind();
    tex_tag         .unbind();
    tex_mass        .unbind();
    tex_rho         .unbind();
    tex_coord_merged.unbind();
    tex_veloc_merged.unbind();

    for(std::map<std::string,TextureObject>::iterator iter = tex_misc_.begin();
    	iter != tex_misc_.end();
    	iter++ ) {
    	iter->second.unbind();
    }
}

void MesoAtom::create_avec( const char *style, int narg, char **arg, char *suffix )
{
    Atom::create_avec( style, narg, arg, suffix );
    if( avec->cudable ) meso_avec = dynamic_cast<MesoAtomVec *>( avec );
}

__global__ void gpu_set_map(
    int* __restrict tag,
    uint* __restrict map_array,
    const int n
)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ) atomicMin( map_array + tag[i], i );
}

__global__ void gpu_hash_map(
    int*  __restrict tag,
    uint* __restrict hash_key,
    uint* __restrict hash_val,
    const uint nonce,
    const int nslot,
    const int nall
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < nall ) {
        uint p, now;
        uint t = tag[i];
        uint u = t;
        uint v = nonce;
        do {
            __TEA_core<8>( u, v );
            p = u % nslot;
            now = atomicCAS( hash_key + p, 0, t );
        } while ( now != 0 && now != t );
		atomicMin( hash_val + p, i ); // favor local particle in case of multiple instances (local+ghost)
    }
}

void MesoAtom::map_set_device()
{
    if( map_style == 1 ) {
        if( !map_tag_max ) {
            int max = 0;
            for( int i = 0; i < nlocal; i++ ) max = MAX( max, tag[i] );
            MPI_Allreduce( &max, &map_tag_max, 1, MPI_INT, MPI_MAX, world );
        }

        if( dev_map_array.n_elem() < map_tag_max + 1 )
            dev_map_array.grow( map_tag_max + 1, false, false );

        dev_map_array.set( 0XFFFFFFFF, meso_device->stream() );
        int threads_per_block = meso_device->query_block_size( gpu_set_map );
        gpu_set_map <<< n_block( nlocal + nghost, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            dev_tag,
            dev_map_array,
            nlocal + nghost );
        tex_map_array.bind( dev_map_array );
    } else {
        int nall = atom->nlocal + atom->nghost;
        int nslot = ceiling( nall / hash_load_factor, 1024 );
        if( nslot > hash_table_size ) {
            meso_device->sync_device();
            hash_table_size = nslot;
            dev_hash_key.grow( hash_table_size, false, false );
            dev_hash_val.grow( hash_table_size, false, false );
        }

        dev_hash_key.set( 0, meso_device->stream() );
        dev_hash_val.set( 0xFFFFFFFF, meso_device->stream() );
        int threads_per_block = meso_device->query_block_size( gpu_hash_map );
        gpu_hash_map <<< n_block( nall, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
            dev_tag,
            dev_hash_key,
            dev_hash_val,
            nonce,
            hash_table_size,
            nall );
        tex_hash_key.bind( dev_hash_key );
        tex_hash_val.bind( dev_hash_val );
    }
}

void MesoAtom::transfer_pre_sort()
{
    // there is no need to sync with any previous GPU events
    // because this transfer only depends on data on the CPU
    // which is surly to be ready when pre_sort(post_exchange) is called

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->pre_sort, CUDACOPY_C2G );
    for( int i = 0 ; i < events.size() ; i++ ) meso_device->stream().waiton( events[i] );
}

void MesoAtom::transfer_pre_post_sort()
{
    // there is no need to sync with any previous GPU events
    // because this transfer only depends on data on the CPU
    // which is surly to be ready when pre_sort(post_exchange) is called

    std::vector<CUDAEvent> events = meso_avec->transfer(
                                   meso_avec->post_sort, CUDACOPY_C2G, dev_permute_from, dev_permute_to, ACTION_COPY );
    // no later work except for the next transfer_post_sort need to wait on this, which by default wait
    // on all previous work
}

void MesoAtom::transfer_post_sort()
{
    CUDAEvent prev_work = meso_device->event( "MesoAtom::previous" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );

    std::vector<CUDAEvent> events = meso_avec->transfer(
                                   meso_avec->post_sort, CUDACOPY_C2G, dev_permute_from, dev_permute_to, ACTION_PERM );
    for( int i = 0 ; i < events.size() ; i++ ) meso_device->stream().waiton( events[i] );
}

void MesoAtom::transfer_pre_border()
{
    CUDAEvent prev_work = meso_device->event( "MesoAtom::previous" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->pre_border, CUDACOPY_G2C );
    meso_device->sync_device();
}

void MesoAtom::transfer_post_border()
{
    // there is no need to sync with any previous GPU events
    // because this transfer only depends on data on the CPU
    // which is surly to be ready when post_border is called

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->post_border, CUDACOPY_C2G );
    for( int i = 0 ; i < events.size() ; i++ ) meso_device->stream().waiton( events[i] );
}

void MesoAtom::transfer_pre_comm()
{
    CUDAEvent prev_work = meso_device->event( "MesoAtom::previous" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->pre_comm, CUDACOPY_G2C );
    for( int i = 0 ; i < events.size() ; i++ ) meso_device->stream( 1 ).waiton( events[i] );
    // this event is used by the integrate style to ensure data has been sent
    // from GPU to CPU before it calls the forward_comm function on the CPU
    meso_device->event( "meso_atom::transfer_pre_comm" ).record( meso_device->stream( 1 ) );
}

void MesoAtom::transfer_post_comm()
{
    // there is no need to sync with any previous GPU events
    // because this transfer only depends on data on the CPU
    // which is surly to be ready when post_comm is called

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->post_comm, CUDACOPY_C2G );
    for( int i = 0 ; i < events.size() ; i++ ) meso_device->stream().waiton( events[i] );
}

void MesoAtom::transfer_pre_exchange()
{
    CUDAEvent prev_work = meso_device->event( "MesoAtom::previous" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );

    static ThreadTuner &o = meso_device->tuner( "MesoAtomVec::transfer", 1, 2 );
    CUDAEvent e( true );
    double t_cpu_0, t_cpu_1;
    if( meso_device->warmed_up() && false && o.lv() < 1 ) {
        e.record( meso_device->stream() );
        t_cpu_0 = meso_device->get_time_omp();
    }

    std::vector<CUDAEvent> events = meso_avec->transfer( meso_avec->pre_exchange, CUDACOPY_G2C );

    if( meso_device->warmed_up() && false && o.lv() < 1 ) {
        t_cpu_1 = meso_device->get_time_omp();
    }

    meso_device->sync_device();

    if( meso_device->warmed_up() && false && o.lv() < 1 ) {
        double tmax = 0.0;
        for( int i = 0 ; i < events.size() ; i++ ) tmax = std::max<double>( tmax, events[i] - e );
        tmax = std::max<double>( tmax, t_cpu_1 - t_cpu_0 );
        o.learn( o.bet(), tmax );
    }
}

void MesoAtom::transfer_pre_output()
{
    CUDAEvent prev_work = meso_device->event( "MesoAtom::previous" );
    prev_work.record( meso_device->stream() );
    CUDAStream::all_waiton( prev_work );

    meso_avec->transfer( meso_avec->pre_output, CUDACOPY_G2C );
    meso_device->sync_device();
}

template<int SORT>
__global__ void gpu_build_reorder_keypair(
    r64 * __restrict coord_x,
    r64 * __restrict coord_y,
    r64 * __restrict coord_z,
    u64 * __restrict key,
    int * __restrict val,
    int * __restrict borderness,
    const double3 my_box_lo,
    const double3 my_box_hi,
    const double3 bin_inv,
    const double3 coord_inv,
    const double3 bin_size,
    const int3 bin_num,
    const int l2_resoln,
    const int l1_shift,
    const int l2_shift,
    const u64 border_mask,
    const int nall )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;

    if( i < nall ) {
        if( SORT ) {
            uint bin_id_x = clamp( ( coord_x[i] - my_box_lo.x ) * bin_inv.x + 1, 0, bin_num.x ) ;
            uint bin_id_y = clamp( ( coord_y[i] - my_box_lo.y ) * bin_inv.y + 1, 0, bin_num.y ) ;
            uint bin_id_z = clamp( ( coord_z[i] - my_box_lo.z ) * bin_inv.z + 1, 0, bin_num.z ) ;
            uint x = clamp( ( coord_x[i] - ( bin_id_x - 1 ) * bin_size.x ) * coord_inv.x, 0, l2_resoln );
            uint y = clamp( ( coord_y[i] - ( bin_id_y - 1 ) * bin_size.y ) * coord_inv.y, 0, l2_resoln );
            uint z = clamp( ( coord_z[i] - ( bin_id_z - 1 ) * bin_size.z ) * coord_inv.z, 0, l2_resoln );
            u64 z1 = morton_encode( bin_id_x, bin_id_y, bin_id_z );
            u64 z2 = morton_encode( x, y, z );
            key[i] = ( z1 << l1_shift ) | ( z2 << l2_shift ) ;
            // border particles always come after bulk particles
            if( borderness[i] ) key[i] |= border_mask;
        } else {
            key[i] = i ; //equiv to turning off sorting
        }
        val[i] = i ;
    }
}

__global__ void gpu_permute_from2to( int * __restrict A, int * __restrict T, const int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i < n ) T[ A[i] ] = i;
}

void MesoAtom::count_bulk_and_border( const std::vector<int> &borderness )
{
    int nb = 0;
    if( borderness.size() ) {
        static ThreadTuner &o = meso_device->tuner( "MesoAtom::count_bulk_and_border" );
        size_t ntd = o.bet();
        double t1 = meso_device->get_time_omp();

        #pragma omp parallel reduction(+:nb)
        {
            int tid = omp_get_thread_num();
            if( tid < ntd ) {
                int2 work = split_work( 0, nlocal, tid, ntd );
                for( int i = work.x; i < work.y; i++ ) if( borderness[i] ) nb++;
            }
        }

        double t2 = meso_device->get_time_omp();
        if( meso_device->warmed_up() ) o.learn( ntd, t2 - t1 );
    } else {
        nb = nlocal;
    }

    n_border = nb;
    n_bulk = nlocal - n_border;
}

void MesoAtom::sort_local()
{
    int l2_resoln     = 16; // fixed to 4-bit per dimension
    double3 bin_size  = make_double3( neighbor->binsizex, neighbor->binsizey, neighbor->binsizez );
    double3 bin_inv   = make_double3( neighbor->bininvx, neighbor->bininvy, neighbor->bininvz );
    double3 coord_inv = make_double3( l2_resoln * bin_inv.x, l2_resoln * bin_inv.y, l2_resoln * bin_inv.z );
    double3 my_box_hi = make_double3( domain->subhi[0], domain->subhi[1], domain->subhi[2] );
    double3 my_box_lo = make_double3( domain->sublo[0], domain->sublo[1], domain->sublo[2] );
    int3    bin_num   = make_int3( neighbor->mbinx, neighbor->mbiny, neighbor->mbinz );
    int     max_bin   = max( max( neighbor->mbinx, neighbor->mbiny ), neighbor->mbinz );
    int    l0_width   = 1;
    int    l1_width   = 3 * floor( log2( max_bin * 2.0 ) );
    int    l2_width   = 3 * log2( ( double )l2_resoln );
    int    l0_shift   = l2_width + l1_width;
    int    l1_shift   = l2_width ;
    int    l2_shift   = 0;
    u64   border_mask = 1ULL << l0_shift;
    assert( l0_width + l1_width + l2_width < 64 );

    meso_comm->borderness( borderness );
    ( *hst_borderness ).copy( borderness );
    count_bulk_and_border( borderness );

    int threads_per_block = meso_device->query_block_size( gpu_build_reorder_keypair<1> );
    gpu_build_reorder_keypair<1> <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        dev_coord(0), dev_coord(1), dev_coord(2),
        dev_perm_key, dev_permute_from,
        hst_borderness,
        my_box_lo, my_box_hi,
        bin_inv, coord_inv, bin_size, bin_num,
        l2_resoln,
        l1_shift, l2_shift,
        border_mask,
        atom->nlocal );

	sorter.sort( *dev_perm_key, *dev_permute_from, atom->nlocal, l0_width + l1_width + l2_width, meso_device->stream() );

    threads_per_block = meso_device->query_block_size( gpu_permute_from2to );
    gpu_permute_from2to <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        dev_permute_from, dev_permute_to,
        atom->nlocal );
}


/* ----------------------------------------------------------------------
     unpack n lines from Atom section of data file
     call style-specific routine to parse line
------------------------------------------------------------------------- */

//struct DRecord
//{
//  int imagedata;
//  double xdata[3],lamda[3];
//  std::vector<char*> values;
//  DRecord()
//  {
//      xdata[0] = xdata[1] = xdata[2] = NULL;
//      lamda[0] = lamda[1] = lamda[2] = NULL;
//      values.clear();
//  }
//  DRecord( int n )
//  {
//      xdata[0] = xdata[1] = xdata[2] = NULL;
//      lamda[0] = lamda[1] = lamda[2] = NULL;
//      values.resize( n );
//  }
//  DRecord( const DRecord &other )
//  {
//      imagedata = other.imagedata;
//      for(int i=0;i<3;i++) xdata[i] = other.xdata[i], lamda[i] = other.lamda[i];
//      values = other.values;
//  }
//  DRecord& operator = ( const DRecord &other )
//  {
//      imagedata = other.imagedata;
//      for(int i=0;i<3;i++) xdata[i] = other.xdata[i], lamda[i] = other.lamda[i];
//      values = other.values;
//      return *this;
//  }
//};
//
//void MesoAtom::data_atoms(int n, char *buf)
//{
////    return Atom::data_atoms(n,buf);
//
//  char *next = strchr(buf,'\n');
//  *next = '\0';
//  int nwords = count_words(buf);
//  *next = '\n';
//
//  if (nwords != avec->size_data_atom && nwords != avec->size_data_atom + 3)
//      error->all(__FILE__,__LINE__,"Incorrect atom format in data file");
//
//  // set bounds for my proc
//  // if periodic and I am lo/hi proc, adjust bounds by EPSILON
//  // insures all data atoms will be owned even with round-off
//
//  double sublo[3],subhi[3];
//  int triclinic = domain->triclinic;
//  if (triclinic == 0)
//  {
//      sublo[0] = domain->sublo[0]; subhi[0] = domain->subhi[0];
//      sublo[1] = domain->sublo[1]; subhi[1] = domain->subhi[1];
//      sublo[2] = domain->sublo[2]; subhi[2] = domain->subhi[2];
//  }
//  else
//  {
//      sublo[0] = domain->sublo_lamda[0]; subhi[0] = domain->subhi_lamda[0];
//      sublo[1] = domain->sublo_lamda[1]; subhi[1] = domain->subhi_lamda[1];
//      sublo[2] = domain->sublo_lamda[2]; subhi[2] = domain->subhi_lamda[2];
//  }
//  double boxlo[3], boxhi[3];
//  boxlo[0] = domain->boxlo[0];
//  boxlo[1] = domain->boxlo[1];
//  boxlo[2] = domain->boxlo[2];
//  boxhi[0] = domain->boxhi[0];
//  boxhi[1] = domain->boxhi[1];
//  boxhi[2] = domain->boxhi[2];
//
//  if (domain->xperiodic)
//  {
//      if (comm->myloc[0] == 0) sublo[0] -= EPSILON;
//      if (comm->myloc[0] == comm->procgrid[0]-1) subhi[0] += EPSILON;
//  }
//  if (domain->yperiodic)
//  {
//      if (comm->myloc[1] == 0) sublo[1] -= EPSILON;
//      if (comm->myloc[1] == comm->procgrid[1]-1) subhi[1] += EPSILON;
//  }
//  if (domain->zperiodic)
//  {
//      if (comm->myloc[2] == 0) sublo[2] -= EPSILON;
//      if (comm->myloc[2] == comm->procgrid[2]-1) subhi[2] += EPSILON;
//  }
//
//  // xptr = which word in line starts xyz coords
//  // iptr = which word in line starts ix,iy,iz image flags
//  int xptr = avec->xcol_data - 1;
//  int imageflag = ( nwords > avec->size_data_atom ) ? 1 : 0;
//  int iptr = imageflag ? ( nwords - 3 ) : 0;
//
//  // loop over lines of atom data
//  // tokenize the line into values
//  // extract xyz coords and image flags
//  // remap atom into simulation box
//  // if atom is in my sub-domain, unpack its values
//
//  std::vector<char*> lines;
//  std::vector<DRecord> records( n, DRecord(nwords) );
//  std::vector<int> accept_flag( n, 0 );
//  std::vector<int> accept_list;
//
//  for (int i = 0; i < n; i++)
//  {
//      next = strchr(buf,'\n');
//      lines.push_back(buf);
//      buf = next + 1;
//  }
//
//  while( atom->nlocal + n > avec->get_nmax() ) avec->grow(0);
//
//  #pragma omp parallel
//  {
//      #pragma omp for
//      for (int i = 0; i < n; i++ )
//      {
//          char   *save;
//          double *coord;
//          DRecord rec( nwords );
//          buf  = lines[i];
//
//          rec.values[0] = strtok_r(buf," \t\n\r\f",&save);
//          for ( int m = 1; m < nwords; m++ )
//              rec.values[m] = strtok_r(NULL," \t\n\r\f",&save);
//
//          if (imageflag)
//              rec.imagedata = ((atoi(rec.values[iptr+2]) + 512 & 1023) << 20) |
//                              ((atoi(rec.values[iptr+1]) + 512 & 1023) << 10) |
//                               (atoi(rec.values[iptr  ]) + 512 & 1023);
//          else rec.imagedata = (512 << 20) | (512 << 10) | 512;
//
//          rec.xdata[0] = atof(rec.values[xptr]);
//          rec.xdata[1] = atof(rec.values[xptr+1]);
//          rec.xdata[2] = atof(rec.values[xptr+2]);
//          domain->remap(rec.xdata,rec.imagedata);
//          if (triclinic)
//          {
//              domain->x2lamda(rec.xdata,rec.lamda);
//              coord = rec.lamda;
//          }
//          else coord = rec.xdata;
//
//          if ( coord[0] >= sublo[0] && coord[0] < subhi[0] &&
//               coord[1] >= sublo[1] && coord[1] < subhi[1] &&
//               coord[2] >= sublo[2] && coord[2] < subhi[2] )
//          {
//              accept_flag[i] = 1;
//          }
//          else
//          {
//              if ( coord[0] < boxlo[0] || coord[0] >= boxhi[0] ||
//                   coord[1] < boxlo[1] || coord[1] >= boxhi[1] ||
//                   coord[2] < boxlo[2] || coord[2] >= boxhi[2] )
//              fprintf(stderr,"particle rejected: %.16lf %.16lf %.16lf\n", coord[0], coord[1], coord[2] );
//          }
//
//          records[i] = rec;
//      }
//
//      #pragma omp single
//          for( int i = 0 ; i < n ; i++ )
//              if ( accept_flag[i] ) accept_list.push_back( i );
//
//      #pragma omp for
//      for( int p = 0 ; p < accept_list.size() ; p++ )
//      {
//          int i = accept_list[p];
//          cuda_atom->cuda_avec->data_atom_target( atom->nlocal+p, records[i].xdata, records[i].imagedata, &(records[i].values[0]) );
//      }
//  }
//
//  atom->nlocal += accept_list.size();
//}

