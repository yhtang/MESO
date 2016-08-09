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
#include "pair_edpd_trp_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

MesoPairEDPDTRP::MesoPairEDPDTRP( LAMMPS *lmp ) : MesoPairEDPDTRPBase( lmp ), MesoPointers( lmp )
{
    split_flag  = 1;
    random = NULL;
}

MesoPairEDPDTRP::~MesoPairEDPDTRP()
{
    if( allocated ) {
        memory->destroy( setflag );
        memory->destroy( cut );
        memory->destroy( cutsq );
        memory->destroy( cut_inv );
        memory->destroy( a0 );
        memory->destroy( gamma );
        memory->destroy( cv );
        memory->destroy( kappa );
        memory->destroy( theta );
        memory->destroy( da );
        memory->destroy( omega );
    }
}

void MesoPairEDPDTRP::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair:setflag" );
    memory->create( cut,     n + 1, n + 1, "pair:cut" );
    memory->create( cutsq,   n + 1, n + 1, "pair:cutsq" );
    memory->create( cut_inv, n + 1, n + 1, "pair:cut_inv" );
    memory->create( a0,      n + 1, n + 1, "pair:a0" );
    memory->create( gamma,   n + 1, n + 1, "pair:gamma" );
    memory->create( cv,      n + 1, n + 1, "pair:cv" );
    memory->create( kappa,   n + 1, n + 1, "pair:kappa" );
    memory->create( theta,   n + 1, n + 1, "pair:theta" );
    memory->create( da,      n + 1, n + 1, "pair:da" );
    memory->create( omega,   n + 1, n + 1, "pair:omega" );

    for( int i = 1; i <= n; i++ )
        for( int j = i; j <= n; j++ )
            setflag[i][j] = 0;

    dev_coefficients.grow( n * n * n_coeff );
}

void MesoPairEDPDTRP::prepare_coeff()
{
    if( coeff_ready ) return;
    if( !allocated ) allocate();

    int n = atom->ntypes;
    static std::vector<r64> coeff_table;
    coeff_table.resize( n * n * n_coeff );
    for( int i = 1; i <= n; i++ ) {
        for( int j = 1; j <= n; j++ ) {
            int cid = ( i - 1 ) * n + ( j - 1 );
            coeff_table[ cid * n_coeff + p_cut   ] = cut[i][j];
            coeff_table[ cid * n_coeff + p_cutinv] = cut_inv[i][j];
            coeff_table[ cid * n_coeff + p_a0    ] = a0[i][j];
            coeff_table[ cid * n_coeff + p_gamma ] = gamma[i][j];
            coeff_table[ cid * n_coeff + p_cv    ] = cv[i][j];
            coeff_table[ cid * n_coeff + p_sq2kpa] = std::sqrt( 2.0 * kappa[i][j] );
            coeff_table[ cid * n_coeff + p_theta ] = theta[i][j];
            coeff_table[ cid * n_coeff + p_da    ] = da[i][j];
            coeff_table[ cid * n_coeff + p_omega ] = omega[i][j];
        }
    }
    dev_coefficients.upload( &coeff_table[0], coeff_table.size(), meso_device->stream() );
    coeff_ready = true;
}

template<int evflag>
__global__ void gpu_pnipam(
    texobj tex_coord, texobj tex_veloc, texobj tex_therm,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    r64* __restrict Q,
    r64* __restrict virial_xx, r64* __restrict virial_yy, r64* __restrict virial_zz,
    r64* __restrict virial_xy, r64* __restrict virial_xz, r64* __restrict virial_yz,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair,
    r64* __restrict coefficients,
    const r64 dt_inv_sqrt,
    const int pair_padding,
    const int n_type,
    const int p_beg,
    const int p_end
)
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int iter = blockIdx.x * blockDim.x + threadIdx.x; ; iter += gridDim.x * blockDim.x ) {
        int i = ( p_beg & WARPALIGN ) + iter;
        if( i >= p_end ) break;
        if( i >= p_beg ) {
            f3u  coord1 = tex1Dfetch<float4>( tex_coord, i );
            f3u  veloc1 = tex1Dfetch<float4>( tex_veloc,  i );
            tmm  therm1 = tex1Dfetch<float4>( tex_therm,  i );
            r64  T1_inv = __rcp( therm1.T );

            int  n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() ) * pair_padding + __laneid();
            r64 fx   = 0., fy   = 0., fz   = 0.;
            r64 q = 0.;
            r64 vrxx = 0., vryy = 0., vrzz = 0.;
            r64 vrxy = 0., vrxz = 0., vryz = 0.;
            r64 energy = 0.;

            for( int p = 0; p < n_pair; p++ ) {
                int j   = __lds( p_pair );
                p_pair += pair_padding;
                if( ( p & 31 ) == 31 ) p_pair -= 32 * pair_padding - 32;

                f3u coord2   = tex1Dfetch<float4>( tex_coord, j );
                r64 dx       = coord1.x - coord2.x;
                r64 dy       = coord1.y - coord2.y;
                r64 dz       = coord1.z - coord2.z;
                r64 rsq      = dx * dx + dy * dy + dz * dz;
                r64 *coeff_ij = coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff;

                if( rsq < power<2>( coeff_ij[p_cut] ) && rsq >= EPSILON_SQ ) {
                    f3u veloc2   = tex1Dfetch<float4>( tex_veloc, j );
                    r64 rinv     = rsqrt( rsq );
                    r64 r        = rsq * rinv;
                    r64 dvx      = veloc1.x - veloc2.x;
                    r64 dvy      = veloc1.y - veloc2.y;
                    r64 dvz      = veloc1.z - veloc2.z;
                    r64 dot      = dx * dvx + dy * dvy + dz * dvz;
                    r64 dot_rinv = dot * rinv;
                    r64 wc       = 1.0 - r * coeff_ij[p_cutinv];
                    r64 wr       = wc;
                    r64 wd       = wr * wr;

                    tmm therm2   = tex1Dfetch<float4>( tex_therm, j );
                    r64 T2_inv   = __rcp( therm2.T );
                    r64 T_ij     = 0.5 * ( therm1.T + therm2.T );
                    r64 gamma_ij = coeff_ij[p_gamma];
                    r64 sigma_ij = 2.0 * sqrt( gamma_ij * __rcp( T1_inv + T2_inv ) );

                    // force
                    {
                        r64 rn       = gaussian_TEA<4>( veloc1.i > veloc2.i, veloc1.i, veloc2.i );
                        r64 alpha_ij = coeff_ij[p_a0] * T_ij; // DPD classical term
                        r64 d_alpha  = coeff_ij[p_da] / ( 1.0 + expf( coeff_ij[p_omega] * ( T_ij - coeff_ij[p_theta]  ) ) ); // temperature-dependence

                        r64 fpair  =  ( alpha_ij + d_alpha ) * wc
                                      - ( gamma_ij * wd * dot_rinv )
                                      + ( sigma_ij * wr * rn * dt_inv_sqrt );
                        fpair     *= rinv;

                        fx += dx * fpair;
                        fy += dy * fpair;
                        fz += dz * fpair;
                        if( evflag ) {
                            energy += 0.5 * alpha_ij * coeff_ij[p_cut] * wc * wc;
                            vrxx += dx * dx * fpair;
                            vryy += dy * dy * fpair;
                            vrzz += dz * dz * fpair;
                            vrxy += dx * dy * fpair;
                            vrxz += dx * dz * fpair;
                            vryz += dy * dz * fpair;
                        }
                    }

                    // heat
                    {
                        r64 wr_t  = wc;
                        r64 wc_t  = wr_t * wr_t;
                        r64 cv    = coeff_ij[p_cv];
                        r64 beta  = coeff_ij[p_sq2kpa] * cv * T_ij;
                        r64 kij   = beta * beta * 0.5;

                        r64 rn2      = gaussian_TEA<4>( therm1.i < therm2.i, therm1.i, therm2.i );
                        r64 mass_inv = sqrtf( therm1.mass_inv * therm2.mass_inv );

                        r64 qpair    = kij * wc_t * ( T1_inv - T2_inv )
                                       + 0.5 * __rcp( cv ) * ( wd * ( gamma_ij * dot_rinv * dot_rinv - sigma_ij * sigma_ij * mass_inv ) - sigma_ij * wr * dot_rinv * rn2 )
                                       + beta * wr_t * dt_inv_sqrt * rn2;

                        q += qpair;
                    }
                }
            }

            force_x[i] += fx;
            force_y[i] += fy;
            force_z[i] += fz;
            Q      [i] += q;
            if( evflag ) {
                e_pair[i] = energy * 0.5;
                virial_xx[i] += vrxx * 0.5;
                virial_yy[i] += vryy * 0.5;
                virial_zz[i] += vrzz * 0.5;
                virial_xy[i] += vrxy * 0.5;
                virial_xz[i] += vrxz * 0.5;
                virial_yz[i] += vryz * 0.5;
            }
        }
    }
}

void MesoPairEDPDTRP::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    prepare_coeff();
    MesoNeighList *dlist = meso_neighbor->lists_device[ list->index ];

    int shared_mem_size = atom->ntypes * atom->ntypes * n_coeff * sizeof( r64 );

    if( eflag || vflag ) {
        // evaluate force, energy and virial
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_pnipam<1>, shared_mem_size );
        gpu_pnipam<1> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged, meso_atom->tex_misc("therm"),
            meso_atom->dev_force [0], meso_atom->dev_force [1], meso_atom->dev_force [2],
            meso_atom->dev_Q,
            meso_atom->dev_virial[0], meso_atom->dev_virial[1], meso_atom->dev_virial[2],
            meso_atom->dev_virial[3], meso_atom->dev_virial[4], meso_atom->dev_virial[5],
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end );
    } else {
        // evaluate force only
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_pnipam<0>, shared_mem_size );
        gpu_pnipam<0> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>> (
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged, meso_atom->tex_misc("therm"),
            meso_atom->dev_force [0], meso_atom->dev_force [1], meso_atom->dev_force [2],
            meso_atom->dev_Q,
            meso_atom->dev_virial[0], meso_atom->dev_virial[1], meso_atom->dev_virial[2],
            meso_atom->dev_virial[3], meso_atom->dev_virial[4], meso_atom->dev_virial[5],
            dlist->dev_pair_count_core, dlist->dev_pair_table,
            meso_atom->dev_e_pair, dev_coefficients,
            1.0 / sqrt( update->dt ), dlist->n_col,
            atom->ntypes, p_beg, p_end );
    }
}

void MesoPairEDPDTRP::compute_bulk( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BULK, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairEDPDTRP::compute_border( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BORDER, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::GHOST, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairEDPDTRP::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

uint MesoPairEDPDTRP::seed_now() {
	return premix_TEA<64>( seed, update->ntimestep );
}

void MesoPairEDPDTRP::settings( int narg, char **arg )
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

void MesoPairEDPDTRP::coeff( int narg, char **arg )
{
    if( narg < 7 )
        error->all( FLERR, "Incorrect args for pair coefficients" );
    if( !allocated ) allocate();

    int ilo, ihi, jlo, jhi;
    force->bounds( arg[0], atom->ntypes, ilo, ihi );
    force->bounds( arg[1], atom->ntypes, jlo, jhi );

    double a0_one    = atof( arg[2] );
    double gamma_one = atof( arg[3] );
    double cut_one   = atof( arg[4] );
    double cv_one    = atof( arg[5] );
    double kappa_one = atof( arg[6] );
    double theta_one = 1.0;
    double da_one    = 0.0;
    double omega_one = 1.0;
    if (narg>7) {
    	theta_one = atof( arg[7] );
    	da_one    = atof( arg[8] );
    	omega_one = atof( arg[9] );
    }

    int count = 0;
    for( int i = ilo; i <= ihi; i++ ) {
        for( int j = MAX( jlo, i ); j <= jhi; j++ ) {
            a0[i][j]    = a0_one;
            gamma[i][j] = gamma_one;
            cut[i][j]   = cut_one;
            cutsq[i][j] = cut_one * cut_one;
            cv[i][j]    = cv_one;
            kappa[i][j] = kappa_one;
            theta[i][j] = theta_one;
            da[i][j]    = da_one;
            omega[i][j] = omega_one;

            if (comm->me == 0)
            	printf("%d %d: a0 %lf gamma %lf cut %lf cv %lf kappa %lf theta %lf da %lf omega %lf\n", i, j, a0_one, gamma_one, cut_one, cv_one, kappa_one, theta_one,da_one, omega_one );

            cut_inv[i][j] = 1.0 / cut_one;
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

void MesoPairEDPDTRP::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton  = 2;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairEDPDTRP::init_one( int i, int j )
{
    if( setflag[i][j] == 0 )
        error->all( FLERR, "All pair coeffs are not set" );

    cut[j][i]     = cut[i][j];
    cut_inv[j][i] = cut_inv[i][j];
    a0[j][i]      = a0[i][j];
    gamma[j][i]   = gamma[i][j];
    cv[j][i]      = cv[i][j];
    kappa[j][i]   = kappa[i][j];
    theta[j][i]   = theta[i][j];
    da   [j][i]   = da   [i][j];
    omega[j][i]   = omega[i][j];

    return cut[i][j];
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairEDPDTRP::write_restart( FILE *fp )
{
    write_restart_settings( fp );

    for( int i = 1; i <= atom->ntypes; i++ ) {
        for( int j = i; j <= atom->ntypes; j++ ) {
            fwrite( &setflag[i][j], sizeof( int ), 1, fp );
            if( setflag[i][j] ) {
                fwrite( &a0[i][j], sizeof( double ), 1, fp );
                fwrite( &gamma[i][j], sizeof( double ), 1, fp );
                fwrite( &cut[i][j], sizeof( double ), 1, fp );
                fwrite( &cv[i][j], sizeof( double ), 1, fp );
                fwrite( &kappa[i][j], sizeof( double ), 1, fp );
                fwrite( &theta[i][j], sizeof( double ), 1, fp );
                fwrite( &da[i][j], sizeof( double ), 1, fp );
                fwrite( &omega[i][j], sizeof( double ), 1, fp );
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairEDPDTRP::read_restart( FILE *fp )
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
                    fread( &a0[i][j], sizeof( double ), 1, fp );
                    fread( &gamma[i][j], sizeof( double ), 1, fp );
                    fread( &cut[i][j], sizeof( double ), 1, fp );
                    fread( &cv[i][j], sizeof( double ), 1, fp );
                    fread( &kappa[i][j], sizeof( double ), 1, fp );
                    fread( &theta[i][j], sizeof( double ), 1, fp );
                    fread( &da[i][j], sizeof( double ), 1, fp );
                    fread( &omega[i][j], sizeof( double ), 1, fp );
                }
                MPI_Bcast( &a0[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &gamma[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &cut[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &cv[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &kappa[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &theta[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &da[i][j], 1, MPI_DOUBLE, 0, world );
                MPI_Bcast( &omega[i][j], 1, MPI_DOUBLE, 0, world );
                cut_inv[i][j] = 1.0 / cut[i][j];
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairEDPDTRP::write_restart_settings( FILE *fp )
{
    fwrite( &cut_global, sizeof( double ), 1, fp );
    fwrite( &seed, sizeof( int ), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairEDPDTRP::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
        fread( &cut_global, sizeof( double ), 1, fp );
        fread( &seed, sizeof( int ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &cut_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &seed, 1, MPI_INT, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );

    if( random ) delete random;
    random = new RanMars( lmp, seed % 899999999 + 1 );
}

/* ---------------------------------------------------------------------- */

double MesoPairEDPDTRP::single( int i, int j, int itype, int jtype, double rsq,
                               double factor_coul, double factor_dpd, double &fforce )
{
    error->warning( FLERR, "<MESO> MesoPairEDPDTRP::single not implemented" );
    return 0.;
}
