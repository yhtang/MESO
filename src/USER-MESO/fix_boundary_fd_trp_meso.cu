/* ----------------------------------------------------------------------
     LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
     http://lammps.sandia.gov, Sandia National Laboratories
     Steve Plimpton, sjplimp@sandia.gov

     Copyright (2003) Sandia Corporation.   Under the terms of Contract
     DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
     certain rights in this software.   This software is distributed under
     the GNU General Public License.

     See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "input.h"
#include "variable.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "fix_boundary_fd_trp_meso.h"
#include "pair_edpd_trp_base_meso.h"

using namespace LAMMPS_NS;
using namespace PNIPAM_COEFFICIENTS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryFdTRP::MesoFixBoundaryFdTRP( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    wall_type(1),
    nx( 0. ), ny( 0. ), nz( 0. ),
    H( 0. ),
    cut( 0. ),
    v0x( 0. ), v0y( 0. ), v0z( 0. ),
    A0( 0. ),
    poly( lmp, "MesoFixBoundaryFdTRP::poly" ),
	pair(NULL)
{
    pair = dynamic_cast<MesoPairEDPDTRPBase*>( force->pair );
    if( !pair ) error->all( FLERR, "<MESO> fix boundary/fc/trp/meso must be used together with pair edpd/pnipam/meso" );

    bool set_H_by_example_point = false;
    double px, py, pz;

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "type" ) ) {
            wall_type = atoi( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "T0" ) ) {
            T0 = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "H" ) ) {
            H = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "v0" ) ) {
            v0x = atof( arg[++i] );
            v0y = atof( arg[++i] );
            v0z = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "n" ) ) {
            nx = atof( arg[++i] );
            ny = atof( arg[++i] );
            nz = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "p" ) ) {
            px = atof( arg[++i] );
            py = atof( arg[++i] );
            pz = atof( arg[++i] );
            set_H_by_example_point = true;
            continue;
        }
        if( !strcmp( arg[i], "A0" ) ) {
            A0 = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "poly" ) ) {
            int order = atoi( arg[++i] );
            poly.grow( order + 1 );
            for( int j = 0; j < order + 1; j++ ) poly[j] = atof( arg[++i] );
            continue;
        }
    }

    if( ( nx == 0. && ny == 0. && nz == 0. ) || poly.n_elem() == 0 || poly == NULL )
        error->all( FLERR, "Usage: boundary/fc group [type int] [T0 double] [cut double] [H double]|[p doublex3] [n doublex3] [poly int doublex?]" );

    double n = std::sqrt( nx * nx + ny * ny + nz * nz );
    nx /= n;
    ny /= n;
    nz /= n;

    if( set_H_by_example_point ) {
        H = px * nx + py * ny + pz * nz;
    }
}

MesoFixBoundaryFdTRP::~MesoFixBoundaryFdTRP()
{
}

int MesoFixBoundaryFdTRP::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryFdTRP::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFdTRP. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryFdTRP::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFdTRP. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_fd_trp(
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    r64* __restrict T,
    int* __restrict type,
    int* __restrict mask,
    r64* __restrict coefficients,
    const int n_type,
    const int wall_type,
    const int groupbit,
    const int order,
    r64* __restrict poly,
    const r64 A0,
    const r64 T0,
    const r64 nx,
    const r64 ny,
    const r64 nz,
    const r64 cut,
    const r64 H,
    const r64 v0x,
    const r64 v0y,
    const r64 v0z,
    const r64 dtinvsqrt,
    const int n )
{
    extern __shared__ r64 coeffs[];
    for( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 d = coord_x[i] * nx + coord_y[i] * ny + coord_z[i] * nz;
            if( d > H - cut ) {
                r64 h         = max( min( H - d, cut ), 0. );
                r64 wc        = max( polyval( h, order, poly ), 0. );
                r64 *coeff_ij = coeffs + ( (type[i]-1) * n_type + (wall_type-1) ) * n_coeff;
				// dissipative force
				r64 gammah    = polyval( h, order, poly );
					gammah   += A0 / h;
					gammah    = max( min( gammah, 1.0 ), 0. );
					gammah   *= coeff_ij[ p_gamma ];
				r64 TT        = 2.0 * T[i] * T0 / ( T[i] + T0 );
				r64 sigmah    = sqrt( 2.0 * TT * gammah );
				r64 vx        = veloc_x[i] - v0x;
				r64 vy        = veloc_y[i] - v0y;
				r64 vz        = veloc_z[i] - v0z;
				r64 v_n_e_n   = vx * nx + vy * ny + vz * nz;
				r64 v_t_x     = vx - v_n_e_n * nx;
				r64 v_t_y     = vy - v_n_e_n * ny;
				r64 v_t_z     = vz - v_n_e_n * nz;
				// balancing random force
				r64 rn        = _SQRT_2 * dtinvsqrt * uniform_TEA<32>( i  , __mantissa( veloc_x[i], veloc_y[i], veloc_z[i] ) );
				// composite force
				r64 v_t       = sqrt( v_t_x*v_t_x + v_t_y*v_t_y + v_t_z*v_t_z );
				if ( v_t > EPSILON ) {
					r64 e_t_x     = v_t_x / v_t;
					r64 e_t_y     = v_t_y / v_t;
					r64 e_t_z     = v_t_z / v_t;
					force_x[i]   -= gammah * v_t_x + sigmah * rn * e_t_x;
					force_y[i]   -= gammah * v_t_y + sigmah * rn * e_t_y;
					force_z[i]   -= gammah * v_t_z + sigmah * rn * e_t_z;
				}
            }
        }
    }
}

void MesoFixBoundaryFdTRP::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_fd_trp, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_fd_trp, cudaFuncCachePreferL1 );
    }

    prepare_coeff();

    double T = T0;
    int var = input->variable->find( (char*)("__PNIPAM_internal_programmable_temperature__") );
    if ( var != -1 ) {
    	T = input->variable->compute_equal(var);
    }

    gpu_fix_boundary_fd_trp <<< grid_cfg.x, grid_cfg.y, pair->dev_coefficients.n_byte(), meso_device->stream() >>> (
		meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
		meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        meso_atom->dev_T,
        meso_atom->dev_type, meso_atom->dev_mask,
        pair->dev_coefficients,
        atom->ntypes,
        wall_type,
        groupbit,
        poly.n_elem() - 1,
        poly,
        A0,
        T,
        nx, ny, nz,
        cut, H,
        v0x, v0y, v0z,
        std::sqrt( 1./update->dt ),
        atom->nlocal );
}

void MesoFixBoundaryFdTRP::prepare_coeff() {
	pair->prepare_coeff();
}
