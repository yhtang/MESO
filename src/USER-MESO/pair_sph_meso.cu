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
#include "pair_sph_meso.h"

using namespace LAMMPS_NS;
using namespace SPH_ONEBODY;
using namespace SPH_TWOBODY;

MesoPairSPH::MesoPairSPH( LAMMPS *lmp ) :
    Pair( lmp ),
    MesoPointers( lmp ),
    dev_coeffs1( lmp, "MesoPairSPHInteraction::dev_coefficients1" ),
    dev_coeffs2( lmp, "MesoPairSPHInteraction::dev_coefficients2" )
{
    split_flag  = 0;
    coeff_ready = false;
}

MesoPairSPH::~MesoPairSPH()
{
    if( allocated ) {
        memory->destroy( setflag );
        memory->destroy( cut );
        memory->destroy( cutsq );
        memory->destroy( rho0 );
        memory->destroy( cs );
        memory->destroy( eta );
    }
}

void MesoPairSPH::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair:setflag" );
    memory->create( cut, n + 1, n + 1, "pair:cut" );
    memory->create( cutsq, n + 1, n + 1, "pair:cutsq" );
    memory->create( rho0, n + 1, "pair:rho0" );
    memory->create( cs, n + 1, "pair:soundspeed" );
    memory->create( cut, n + 1, n + 1, "pair:cut" );
    memory->create( eta, n + 1, n + 1, "pair:viscosity" );

    for( int i = 1; i <= n; i++ )
        for( int j = i; j <= n; j++ )
            setflag[i][j] = 0;
}

void MesoPairSPH::prepare_coeff()
{
    if( coeff_ready ) return;
    if( !allocated ) allocate();

    int n = atom->ntypes;

    coeff_table1.resize( n * n_coeff1 );
    dev_coeffs1.grow( coeff_table1.size() );
    for( int i = 1; i <= n; i++ ) {
        int cid = i - 1;
        coeff_table1[ cid * n_coeff1 + p_rho0_inv ] = 1.0 / rho0[i];
        coeff_table1[ cid * n_coeff1 + p_cs       ] = cs  [i];
        coeff_table1[ cid * n_coeff1 + p_B        ] = B_one( i );
    }
    dev_coeffs1.upload( coeff_table1.data(), n * n_coeff1, meso_device->stream() );

    coeff_table2.resize( n * n * n_coeff2 );
    dev_coeffs2.grow( coeff_table2.size() );
    for( int i = 1; i <= n; i++ ) {
        for( int j = 1; j <= n; j++ ) {
            int cid = ( i - 1 ) * n + ( j - 1 );
            coeff_table2[ cid * n_coeff2 + p_cut    ] = cut[i][j];
            coeff_table2[ cid * n_coeff2 + p_cutinv ] = 1.0 / cut[i][j];
            coeff_table2[ cid * n_coeff2 + p_eta    ] = eta[i][j];
            coeff_table2[ cid * n_coeff2 + p_n3     ] = 1.0 / SPHKernelTang3D<r64>::norm( cut[i][j] );
        }
    }
    dev_coeffs2.upload( coeff_table2.data(), n * n * n_coeff2 , meso_device->stream() );
    coeff_ready = true;
}

__global__ void gpu_sph(
    texobj tex_coord, texobj tex_veloc, texobj tex_rho,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict coefficients1,
    r64* __restrict coefficients2,
    const int pair_padding,
    const int n_type,
    const int p_beg,
    const int p_end,
    const int n_part )
{
    int block_per_part = gridDim.x / n_part;
    int part_id = blockIdx.x / block_per_part;
    if( part_id >= n_part ) return;
    int part_size = block_per_part * blockDim.x;
    int id_in_partition = blockIdx.x % block_per_part * blockDim.x + threadIdx.x;

    extern __shared__ r64 shmem[];
    r64 *coeffs1 = shmem;
    r64 *coeffs2 = shmem + n_type * n_coeff1;
    for( int p = threadIdx.x; p < n_type * n_coeff1; p += blockDim.x )
        coeffs1[p] = coefficients1[p];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff2; p += blockDim.x )
        coeffs2[p] = coefficients2[p];
    __syncthreads();

    for( int iter = id_in_partition; ; iter += part_size ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u    coord1 = tex1Dfetch<float4>( tex_coord, i );
            float4 veloc1 = tex1Dfetch<float4>( tex_veloc,  i );
            r64   *coeff_i = coeffs1 + coord1.i * n_coeff1;
            r64    rho1    = tex1Dfetch<r64>( tex_rho, i );
            r64    m1      = veloc1.w;
            r64    p1      = coeff_i[ p_B ] * ( power<7>( rho1 * coeff_i[ p_rho0_inv ] ) - r64( 1.0 ) );
            r64    fv1     = power<2>( m1 ) * __rcp( power<2>( rho1 ) );
            r64    fc1     = fv1 * p1;
            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() + part_id ) * pair_padding + __laneid();
            r64 fx   = 0., fy   = 0., fz   = 0.;

            for( int p = part_id; p < n_pair; p += n_part ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding * n_part;
                if( ( p & 31 ) + n_part >= WARPSZ ) p_pair -= WARPSZ * pair_padding - WARPSZ;

                f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
                r64 dx       = coord1.x - coord2.x;
                r64 dy       = coord1.y - coord2.y;
                r64 dz       = coord1.z - coord2.z;
                r64 rsq      = dx * dx + dy * dy + dz * dz;
                r64 *coeff_ij = coeffs2 + ( coord1.i * n_type + coord2.i ) * n_coeff2;

                if( rsq < power<2>( coeff_ij[p_cut] ) && rsq >= EPSILON_SQ ) {
                    float4 veloc2 = tex1Dfetch<float4>( tex_veloc, j );
                    r64   *coeff_j = coeffs1 + coord2.i * n_coeff1;
                    r64    rho2    = tex1Dfetch<r64>( tex_rho, j );
                    r64    m2      = veloc2.w;
                    r64    p2      = coeff_j[ p_B ] * ( power<7>( rho2 * coeff_j[ p_rho0_inv ] ) - r64( 1.0 ) );
                    r64    fv2     = power<2>( m2 ) * __rcp( power<2>( rho2 ) );
                    r64    fc2     = fv2 * p2;
                    r64 rinv      = rsqrt( rsq );
                    r64 r         = rsq * rinv;
                    r64 dvx       = veloc1.x - veloc2.x;
                    r64 dvy       = veloc1.y - veloc2.y;
                    r64 dvz       = veloc1.z - veloc2.z;
//                  SPHKernelGauss3D<r64> kernel(coeff_ij[p_cut]);
//                  r64 dW        = kernel.gradient(r);
                    SPHKernelTang3D<r64> kernel( coeff_ij[p_cutinv] );
                    r64 dW        = kernel.gradient( r ) * coeff_ij[p_n3];

                    r64 f_cons  =  -( fc1 + fc2 ) * dW  * rinv;
                    r64 f_visc  =  coeff_ij[ p_eta ] * ( fv1 + fv2 ) * dW  * rinv;

                    fx += dx * f_cons + dvx * f_visc;
                    fy += dy * f_cons + dvy * f_visc;
                    fz += dz * f_cons + dvz * f_visc;
                }
            }

            if( n_part == 1 ) {
                force_x[i] += fx;
                force_y[i] += fy;
                force_z[i] += fz;
            } else {
                atomic_add( force_x + i, fx );
                atomic_add( force_y + i, fy );
                atomic_add( force_z + i, fz );
            }
        }
    }
}

void MesoPairSPH::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    if( !coeff_ready ) prepare_coeff();
    MesoNeighList *dlist = meso_neighbor->lists_device[ list->index ];

    int shared_mem_size = ( atom->ntypes * n_coeff1 + atom->ntypes * atom->ntypes * n_coeff2 ) * sizeof( r64 );

    // evaluate force only
    static GridConfig grid_cfg = meso_device->configure_kernel( gpu_sph, shared_mem_size );
    gpu_sph <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
        meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged, meso_atom->tex_rho,
        meso_atom->dev_force(0),   meso_atom->dev_force(1),   meso_atom->dev_force(2),
        dlist->dev_pair_count_core, dlist->dev_pair_table,
        dev_coeffs1, dev_coeffs2,
        dlist->n_col, atom->ntypes,
        p_beg, p_end, grid_cfg.partition( p_end - p_beg, WARPSZ ) );
}

void MesoPairSPH::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed, c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairSPH::settings( int narg, char **arg )
{
    if( narg < 0 ) error->all( FLERR, "Illegal pair_style command" );

    if( narg > 0 ) cut_global  = atof( arg[0] );
    if( narg > 1 ) rho0_global = atof( arg[1] );
    if( narg > 2 ) cs_global   = atof( arg[2] );
    if( narg > 3 ) eta_global  = atof( arg[3] );

    // reset cutoffs that have been explicitly set
    if( allocated ) {
        for( int i = 1; i <= atom->ntypes; i++ ) {
            for( int j = i + 1; j <= atom->ntypes; j++ ) {
                if( setflag[i][j] ) {
                    cut[i][j] = cut_global;
                    cutsq[i][j] = cut_global * cut_global;
                    rho0[i] = rho0_global;
                    cs[i] = cs_global;
                    eta[i][j] = eta_global;
                }
            }
        }
    }
}

void MesoPairSPH::coeff( int narg, char **arg )
{
    if( narg != 6 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
    if( !allocated ) allocate();

    int ilo, ihi, jlo, jhi;
    force->bounds( arg[0], atom->ntypes, ilo, ihi );
    force->bounds( arg[1], atom->ntypes, jlo, jhi );

    double rho0_one = force->numeric( FLERR, arg[2] );
    double cs_one   = force->numeric( FLERR, arg[3] );
    double eta_one  = force->numeric( FLERR, arg[4] );
    double cut_one  = force->numeric( FLERR, arg[5] );

    int count = 0;
    for( int i = ilo; i <= ihi; i++ ) {
        rho0[i] = rho0_one;
        cs  [i] = cs_one;
        for( int j = MAX( jlo, i ); j <= jhi; j++ ) {
            cut    [i][j] = cut_one;
            cutsq  [i][j] = cut_one * cut_one;
            eta    [i][j] = eta_one;
            setflag[i][j] = 1;
            count++;
        }
    }

    coeff_ready = false;

    if( count == 0 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void MesoPairSPH::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton  = 2;
    neighbor->requests[i]->ghost   = 0;

    char fix_args[3][256];
    strcpy( fix_args[0], "this_id_is_reserved_for_MesoPairSPH" );
    strcpy( fix_args[1], "all" );
    strcpy( fix_args[2], "rho/meso" );
    modify->add_fix( 3, (char**)fix_args, NULL );
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairSPH::init_one( int i, int j )
{
    if( setflag[i][j] == 0 )
        error->all( FLERR, "All pair coeffs are not set" );

    cut    [j][i] = cut[i][j];
    cutsq  [j][i] = cutsq[i][j];
    eta    [j][i] = eta[i][j];

    return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairSPH::write_restart( FILE *fp )
{
    write_restart_settings( fp );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        fwrite( &rho0[i], sizeof( double ), 1, fp );
        fwrite( &cs  [i], sizeof( double ), 1, fp );
        for( int j = i; j <= atom->ntypes; j++ ) {
            fwrite( &setflag[i][j], sizeof( int ), 1, fp );
            if( setflag[i][j] ) {
                fwrite( &cut[i][j], sizeof( double ), 1, fp );
                fwrite( &eta[i][j], sizeof( double ), 1, fp );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairSPH::read_restart( FILE *fp )
{
    read_restart_settings( fp );

    allocate();

    int i, j;
    int me = comm->me;
    for( i = 1; i <= atom->ntypes; i++ ) {
        if( me == 0 ) fread( &rho0[i], sizeof( double ), 1, fp );
        if( me == 0 ) fread( &cs  [i], sizeof( double ), 1, fp );
        MPI_Bcast( &rho0[i], 1, MPI_DOUBLE, 0, world );
        MPI_Bcast( &cs  [i], 1, MPI_DOUBLE, 0, world );
        for( j = i; j <= atom->ntypes; j++ ) {
            if( me == 0 ) fread( &setflag[i][j], sizeof( int ), 1, fp );
            MPI_Bcast( &setflag[i][j], 1, MPI_INT, 0, world );
            if( setflag[i][j] ) {
                if( me == 0 ) {
                    fread( &cut[i][j], sizeof( double ), 1, fp );
                    fread( &eta[i][j], sizeof( double ), 1, fp );
                }
                MPI_Bcast( &cut[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &eta[i][j], 1, MPI_DOUBLE, 0, world );
                cutsq[i][j] = cut[i][j] * cut[i][j];
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairSPH::write_restart_settings( FILE *fp )
{
    fwrite( &cut_global, sizeof( double ), 1, fp );
    fwrite( &rho0_global, sizeof( double ), 1, fp );
    fwrite( &cs_global, sizeof( double ), 1, fp );
    fwrite( &eta_global, sizeof( double ), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairSPH::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
        fread( &cut_global, sizeof( double ), 1, fp );
        fread( &rho0_global, sizeof( double ), 1, fp );
        fread( &cs_global, sizeof( double ), 1, fp );
        fread( &eta_global, sizeof( double ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &cut_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &rho0_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &cs_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &eta_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );
}

/* ---------------------------------------------------------------------- */

double MesoPairSPH::single( int i, int j, int itype, int jtype, double rsq,
                            double factor_coul, double factor_sdpd_interaction, double &fforce )
{
    return 0.0;
}
