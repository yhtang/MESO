#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "atom_vec.h"
#include "update.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "fix.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"
#include "pair_dpd_minimal_meso.h"

using namespace LAMMPS_NS;

MesoPairDPDMini::MesoPairDPDMini( LAMMPS *lmp ) : Pair( lmp ), MesoPointers( lmp )
{
    split_flag  = 1;
    random = NULL;
}

MesoPairDPDMini::~MesoPairDPDMini()
{
    if( allocated ) {
        memory->destroy( setflag );
        memory->destroy( cutsq );
    }
}

void MesoPairDPDMini::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair:setflag" );
    memory->create( cutsq,   n + 1, n + 1, "pair:cutsq" );
    for( int i = 1; i <= n; i++ )
        for( int j = i; j <= n; j++ )
            setflag[i][j] = 0;
}

// floating point version of LCG
__inline__ __device__ float rem( float r ) { return r - floorf( r ); }
// FMA wrapper for the convenience of switching rouding modes
__inline__ __device__ float FMA( float x, float y, float z ) {
    return __fmaf_rz( x, y, z );
}
// logistic rounds
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL
template<int N> __inline__ __device__ float __logistic_core( float x ) {
    float x2 = x * x;
    float r = FMA( FMA( 8.0f, x2, -8.0f ), x2, 1.0f );
    return __logistic_core < N - 2 > ( r );
}

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
    return FMA( 2.0f * x, x, -1.0f );
}

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
    return x;
}

// spacing coefficints for low discrepancy numbers
const static float sqrt2 = 1.41421356237309514547;

// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD
template<int N>
__inline__ __device__ float mean0var1( uint u, uint v )
{
    float p = u / 4294967296.f + v / 4294967296.f - 1.0f;
    float l = __logistic_core<N>( p );
    return l * sqrt2;
}

template<int evflag>
__global__ void gpu_dpd_mini(
    texobj tex_coord, texobj tex_veloc,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    r64* __restrict virial_xx, r64* __restrict virial_yy, r64* __restrict virial_zz,
    r64* __restrict virial_xy, r64* __restrict virial_xz, r64* __restrict virial_yz,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair,
    const r32 a0,
    const r32 gamma,
    const r32 sigma_dt,
    const int pair_padding,
    const int p_beg,
    const int p_end )
{
    for( int iter = blockIdx.x * blockDim.x + threadIdx.x; ; iter += gridDim.x * blockDim.x ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
            f3u  veloc1 = tex1Dfetch<float4>( tex_veloc,  i );
            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
            r32 fx   = 0.f, fy   = 0.f, fz   = 0.f;
            r32 vrxx = 0.f, vryy = 0.f, vrzz = 0.f;
            r32 vrxy = 0.f, vrxz = 0.f, vryz = 0.f;
            r32 energy = 0.f;

            for( int p = 0; p < n_pair; p++ ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding;
                if( ( p & 31 ) == WARPSZ - 1 ) p_pair -= WARPSZ * pair_padding - WARPSZ;

                f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
                r32 dx       = coord1.x - coord2.x;
                r32 dy       = coord1.y - coord2.y;
                r32 dz       = coord1.z - coord2.z;
                r32 rsq      = dx * dx + dy * dy + dz * dz;

                if( rsq < 1.f ) {
                    f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
                    r32 rn       = mean0var1<8>( min(veloc1.i, veloc2.i), max(veloc1.i, veloc2.i) );
                    r32 rinv     = rsqrtf( rsq );
                    r32 r        = rsq * rinv;
                    r32 dvx      = veloc1.x - veloc2.x;
                    r32 dvy      = veloc1.y - veloc2.y;
                    r32 dvz      = veloc1.z - veloc2.z;
                    r32 dot      = dx * dvx + dy * dvy + dz * dvz;
                    r32 wc       = max( 0.f, 1.0f - r );
                    r32 wr       = wc;

                    r32 fpair  =  a0 * wc
                                  - gamma * wr * wr * dot * rinv
                                  + sigma_dt * wr * rn;
                    fpair     *= rinv;

                    fx += dx * fpair;
                    fy += dy * fpair;
                    fz += dz * fpair;

                    if( evflag ) {
                        vrxx += dx * dx * fpair;
                        vryy += dy * dy * fpair;
                        vrzz += dz * dz * fpair;
                        vrxy += dx * dy * fpair;
                        vrxz += dx * dz * fpair;
                        vryz += dy * dz * fpair;
                        energy += 0.5f * a0 * wc * wc;
                    }
                }
            }

			force_x[i] += fx;
			force_y[i] += fy;
			force_z[i] += fz;
			if( evflag ) {
				virial_xx[i] += vrxx * 0.5f;
				virial_yy[i] += vryy * 0.5f;
				virial_zz[i] += vrzz * 0.5f;
				virial_xy[i] += vrxy * 0.5f;
				virial_xz[i] += vrxz * 0.5f;
				virial_yz[i] += vryz * 0.5f;
				e_pair[i] = energy * 0.5f;
			}
        }
    }
}

void MesoPairDPDMini::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    MesoNeighList *dlist = meso_neighbor->lists_device[ list->index ];

    if( eflag || vflag ) {
        // evaluate force, energy and virial
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_dpd_mini<1>, 0 );
        gpu_dpd_mini<1> <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0),   meso_atom->dev_force(1),   meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, a0, gamma, sigma * 1.0 / sqrt( update->dt ),
            dlist->n_col,
            p_beg, p_end );
    } else {
        // evaluate force only
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_dpd_mini<0>, 0 );
        gpu_dpd_mini<0> <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0),   meso_atom->dev_force(1),   meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, a0, gamma, sigma * 1.0 / sqrt( update->dt ),
            dlist->n_col,
            p_beg, p_end );
    }
}

void MesoPairDPDMini::compute_bulk( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BULK, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairDPDMini::compute_border( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BORDER, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::GHOST, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairDPDMini::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

uint MesoPairDPDMini::seed_now() {
	return premix_TEA<64>( seed, update->ntimestep );
}

void MesoPairDPDMini::settings( int narg, char **arg )
{
    if( narg != 2 ) error->all( FLERR, "Illegal pair_style command" );

    seed = atoi( arg[1] );
    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );
}

void MesoPairDPDMini::coeff( int narg, char **arg )
{
    if( narg < 5 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
    if( !allocated ) allocate();

    a0    = atof( arg[2] );
    gamma = atof( arg[3] );
    sigma = atof( arg[4] );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        for( int j = 1; j <= atom->ntypes; j++ ) {
            setflag[i][j] = 1;
            cutsq[i][j] = 1.0;
        }
    }
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void MesoPairDPDMini::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton  = 2;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairDPDMini::init_one( int i, int j )
{
    return 1.0;
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairDPDMini::write_restart( FILE *fp )
{
    write_restart_settings( fp );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        for( int j = i; j <= atom->ntypes; j++ ) {
            fwrite( &setflag[i][j], sizeof( int ), 1, fp );
            if( setflag[i][j] ) {
                fwrite( &a0, sizeof( float ), 1, fp );
                fwrite( &gamma, sizeof( float ), 1, fp );
                fwrite( &sigma, sizeof( float ), 1, fp );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairDPDMini::read_restart( FILE *fp )
{
    read_restart_settings( fp );

    allocate();

    int i, j;
    int me = comm->me;
    for( i = 1; i <= atom->ntypes; i++ ) {
        for( j = i; j <= atom->ntypes; j++ ) {
            if( me == 0 )
                fread( &setflag[i][j], sizeof( int ), 1, fp );
            MPI_Bcast( &setflag[i][j], 1, MPI_INT, 0, world );
            if( setflag[i][j] ) {
                if( me == 0 ) {
                    fread( &a0, sizeof( float ), 1, fp );
                    fread( &gamma, sizeof( float ), 1, fp );
                    fread( &sigma, sizeof( float ), 1, fp );
                }
                MPI_Bcast( &a0, 1, MPI_FLOAT, 0, world );
                MPI_Bcast( &gamma, 1, MPI_FLOAT, 0, world );
                MPI_Bcast( &sigma, 1, MPI_FLOAT, 0, world );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairDPDMini::write_restart_settings( FILE *fp )
{
    fwrite( &seed, sizeof( int ), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairDPDMini::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
        fread( &seed, sizeof( int ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &seed, 1, MPI_INT, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );

    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );
}
