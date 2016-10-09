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
#include "pair_dpd_polyforce_meso.h"

using namespace LAMMPS_NS;
using namespace PEPTOID_COEFFICIENTS;

MesoPairDPDPolyForce::MesoPairDPDPolyForce( LAMMPS *lmp ) : Pair( lmp ), MesoPointers( lmp ),
    dev_coefficients( lmp, "MesoPairDPDPolyForce::dev_coefficients" ),
    dev_polynomial( lmp, "MesoPairDPDPolyForce::dev_polynomial" )
{
    split_flag  = 1;
    coeff_ready = false;
    random = NULL;
}

MesoPairDPDPolyForce::~MesoPairDPDPolyForce()
{
    if( allocated ) {
        memory->destroy( setflag );
        memory->destroy( cutsq );
        memory->destroy( cut );
        memory->destroy( cut_inv );
        memory->destroy( gamma );
        memory->destroy( sigma );
    }
}

void MesoPairDPDPolyForce::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair:setflag" );
    memory->create( cutsq,   n + 1, n + 1, "pair:cutsq" );
    memory->create( cut,     n + 1, n + 1, "pair:cut" );
    memory->create( cut_inv, n + 1, n + 1, "pair:cut_inv" );
    memory->create( gamma,   n + 1, n + 1, "pair:gamma" );
    memory->create( sigma,   n + 1, n + 1, "pair:sigma" );
    for( int i = 1; i <= n; i++ )
        for( int j = i; j <= n; j++ )
            setflag[i][j] = 0;

    polynomial.resize( n * n * ( polynomial_maxlen + 1 ) );
    dev_coefficients.grow( n * n * n_coeff );
    dev_polynomial.grow( n * n * ( polynomial_maxlen + 1 ) );
}

void MesoPairDPDPolyForce::prepare_coeff()
{
    if( coeff_ready ) return;
    if( !allocated ) allocate();

    int n = atom->ntypes;
    coeff_table.resize( n * n * n_coeff );
    for( int i = 1; i <= n; i++ ) {
        for( int j = 1; j <= n; j++ ) {
            int cid = ( i - 1 ) * n + ( j - 1 );
            coeff_table[ cid * n_coeff + p_cut   ] = cut[i][j];
            coeff_table[ cid * n_coeff + p_cutsq ] = cutsq[i][j];
            coeff_table[ cid * n_coeff + p_cutinv] = cut_inv[i][j];
            coeff_table[ cid * n_coeff + p_gamma ] = gamma[i][j];
            coeff_table[ cid * n_coeff + p_sigma ] = sigma[i][j];
        }
    }

    dev_coefficients.upload( &coeff_table[0], coeff_table.size(), meso_device->stream() );
    dev_polynomial.upload( &polynomial[0], polynomial.size(), meso_device->stream() );
    coeff_ready = true;
}

template<int evflag>
__global__ void gpu_dpd_polyforce(
    texobj tex_coord, texobj tex_veloc,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    r64* __restrict virial_xx, r64* __restrict virial_yy, r64* __restrict virial_zz,
    r64* __restrict virial_xy, r64* __restrict virial_xz, r64* __restrict virial_yz,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair,
    r32* __restrict coefficients,
    r32* __restrict polynomial,
    const r32 dt_inv_sqrt,
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

    extern __shared__ r32 smem[];
    r32 *coeffs = smem;
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    r32 *poly = smem + n_type * n_type * n_coeff;
    for( int p = threadIdx.x; p < n_type * n_type * (polynomial_maxlen+1); p += blockDim.x )
        poly[p] = polynomial[p];
    __syncthreads();

    for( int iter = id_in_partition; ; iter += part_size ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
            f3u  veloc1 = tex1Dfetch<float4>( tex_veloc,  i );
            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() + part_id ) * pair_padding + __laneid();
            r32 fx   = 0.f, fy   = 0.f, fz   = 0.f;
            r32 vrxx = 0.f, vryy = 0.f, vrzz = 0.f;
            r32 vrxy = 0.f, vrxz = 0.f, vryz = 0.f;
            r32 energy = 0.f;

            for( int p = part_id; p < n_pair; p += n_part ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding * n_part;
                if( ( p & 31 ) + n_part >= WARPSZ ) p_pair -= WARPSZ * pair_padding - WARPSZ;

                f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
                r32 dx       = coord1.x - coord2.x;
                r32 dy       = coord1.y - coord2.y;
                r32 dz       = coord1.z - coord2.z;
                r32 rsq      = dx * dx + dy * dy + dz * dz;
                r32 *coeff_ij = coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff;

                if( rsq < coeff_ij[p_cutsq] && rsq >= EPSILON_SQ ) {
                    f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
                    r32 rn       = gaussian_TEA_fast<4>( veloc1.i > veloc2.i, veloc1.i, veloc2.i );
                    r32 rinv     = rsqrtf( rsq );
                    r32 r        = rsq * rinv;
                    r32 dvx      = veloc1.x - veloc2.x;
                    r32 dvy      = veloc1.y - veloc2.y;
                    r32 dvz      = veloc1.z - veloc2.z;
                    r32 dot      = dx * dvx + dy * dvy + dz * dvz;
                    r32 rrinv    = r * coeff_ij[p_cutinv];
                    r32 wr       = 1.0f - rrinv;

                    r32 *poly_ij = poly + ( coord1.i * n_type + coord2.i ) * (polynomial_maxlen + 1);
                    r32 fpair    = polyval( 1.0f - rrinv, poly_ij[0], poly_ij + 1 )
                                  - ( coeff_ij[p_gamma] * wr * wr * dot * rinv )
                                  + ( coeff_ij[p_sigma] * wr * rn * dt_inv_sqrt );
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
                        energy += polyval_integral( 1.0f - rrinv, poly_ij[0], poly_ij + 1 );
                    }
                }
            }

            if( n_part == 1 ) {
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
            } else {
                atomic_add( force_x + i, fx );
                atomic_add( force_y + i, fy );
                atomic_add( force_z + i, fz );
                if( evflag ) {
                    atomic_add( virial_xx + i, vrxx * 0.5f );
                    atomic_add( virial_yy + i, vryy * 0.5f );
                    atomic_add( virial_zz + i, vrzz * 0.5f );
                    atomic_add( virial_xy + i, vrxy * 0.5f );
                    atomic_add( virial_yz + i, vryz * 0.5f );
                    atomic_add( virial_xz + i, vrxz * 0.5f );
                    atomic_add( e_pair + i, energy * 0.5f );
                }
            }
        }
    }
}

void MesoPairDPDPolyForce::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    if( !coeff_ready ) prepare_coeff();
    MesoNeighList *dlist = meso_neighbor->lists_device[ list->index ];

    int shared_mem_size = atom->ntypes * atom->ntypes * ( n_coeff + polynomial_maxlen + 1 ) * sizeof( r32 );

    if( eflag || vflag ) {
        // evaluate force, energy and virial
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_dpd_polyforce<1>, shared_mem_size );
        gpu_dpd_polyforce<1> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0),   meso_atom->dev_force(1),   meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients, dev_polynomial,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end, grid_cfg.partition( p_end - p_beg, WARPSZ ) );
    } else {
        // evaluate force only
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_dpd_polyforce<0>, shared_mem_size );
        gpu_dpd_polyforce<0> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0),   meso_atom->dev_force(1),   meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients, dev_polynomial,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end, grid_cfg.partition( p_end - p_beg, WARPSZ ) );
    }
}

void MesoPairDPDPolyForce::compute_bulk( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BULK, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairDPDPolyForce::compute_border( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BORDER, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::GHOST, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairDPDPolyForce::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

uint MesoPairDPDPolyForce::seed_now() {
	return premix_TEA<64>( seed, update->ntimestep );
}

void MesoPairDPDPolyForce::settings( int narg, char **arg )
{
    if( narg != 2 ) error->all( FLERR, "Illegal pair_style command" );

    cut_global = atof( arg[0] );
    seed = atoi( arg[1] );
    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );

    // reset cutoffs that have been explicitly set
    if( allocated ) {
        for( int i = 1; i <= atom->ntypes; i++ )
            for( int j = i + 1; j <= atom->ntypes; j++ )
                if( setflag[i][j] )
                    cut[i][j] = cut_global, cut_inv[i][j] = 1.0 / cut_global;
    }
}

void MesoPairDPDPolyForce::coeff( int narg, char **arg )
{
    if( narg < 6 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
    if( !allocated )
        allocate();

    int ilo, ihi, jlo, jhi;
    force->bounds( arg[0], atom->ntypes, ilo, ihi );
    force->bounds( arg[1], atom->ntypes, jlo, jhi );

    float cut_one = cut_global;
    float gamma_one = atof( arg[2] );
    float sigma_one = atof( arg[3] );

    int count = 0;
    for( int i = ilo; i <= ihi; i++ ) {
        for( int j = MAX( jlo, i ); j <= jhi; j++ ) {
            gamma[i][j] = gamma_one;
            sigma[i][j] = sigma_one;
            cut[i][j] = cut_one;
            cutsq[i][j] = cut_one * cut_one;
            cut_inv[i][j] = 1.0 / cut_one;
            int order = atoi( arg[4] );
            int I = i-1;
            int J = j-1;
            polynomial[ ( I * atom->ntypes + J ) * (polynomial_maxlen+1) ] = order;
            polynomial[ ( J * atom->ntypes + I ) * (polynomial_maxlen+1) ] = order;
            for(int k = 0 ; k <= order ; k++) {
            	polynomial[ ( I * atom->ntypes + J ) * (polynomial_maxlen+1) + k + 1 ] = atof( arg[5+k] );
            	polynomial[ ( J * atom->ntypes + I ) * (polynomial_maxlen+1) + k + 1 ] = atof( arg[5+k] );
            }
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

void MesoPairDPDPolyForce::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton  = 2;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairDPDPolyForce::init_one( int i, int j )
{
    if( setflag[i][j] == 0 )
        error->all( FLERR, "All pair coeffs are not set" );

    cut[j][i]     = cut[i][j];
    cut_inv[j][i] = cut_inv[i][j];
    gamma[j][i]   = gamma[i][j];
    sigma[j][i]   = sigma[i][j];

    return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairDPDPolyForce::write_restart( FILE *fp )
{
    write_restart_settings( fp );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        for( int j = i; j <= atom->ntypes; j++ ) {
            fwrite( &setflag[i][j], sizeof( int ), 1, fp );
            if( setflag[i][j] ) {
                fwrite( &gamma[i][j], sizeof( float ), 1, fp );
                fwrite( &sigma[i][j], sizeof( float ), 1, fp );
                fwrite( &cut[i][j], sizeof( float ), 1, fp );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairDPDPolyForce::read_restart( FILE *fp )
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
                    fread( &gamma[i][j], sizeof( float ), 1, fp );
                    fread( &sigma[i][j], sizeof( float ), 1, fp );
                    fread( &cut[i][j], sizeof( float ), 1, fp );
                }
                MPI_Bcast( &gamma[i][j], 1, MPI_FLOAT, 0, world );
                MPI_Bcast( &sigma[i][j], 1, MPI_FLOAT, 0, world );
                MPI_Bcast( &cut[i][j], 1, MPI_FLOAT, 0, world );
                cut_inv[i][j] = 1.0 / cut[i][j];
            }
        }
    }

    error->warning( FLERR, "PolyForce polynomial not loaded in read_restart, please set using the pair_coeff command");
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairDPDPolyForce::write_restart_settings( FILE *fp )
{
    fwrite( &cut_global, sizeof( float ), 1, fp );
    fwrite( &seed, sizeof( int ), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairDPDPolyForce::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
        fread( &cut_global, sizeof( float ), 1, fp );
        fread( &seed, sizeof( int ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &cut_global, 1, MPI_FLOAT, 0, world );
    MPI_Bcast( &seed, 1, MPI_INT, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );

    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );
}
