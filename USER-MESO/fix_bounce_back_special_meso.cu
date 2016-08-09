#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "domain.h"

#include "atom_meso.h"
#include "comm_meso.h"
#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "fix_bounce_back_special_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixBounceBackSpecial::MesoFixBounceBackSpecial( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    cx( 0. ), cy( 0. ), cz( 0. ),
    ox( 0. ), oy( 0. ), oz( 0. ),
    radius( 0. )
{
    if( narg < 4 ) error->all( FLERR, "Illegal fix bounceback/special/meso command" );

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "radius" ) ) {
            radius = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "center" ) ) {
            cx = atof( arg[++i] );
            cy = atof( arg[++i] );
            cz = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "orient" ) ) {
            ox = atof( arg[++i] );
            oy = atof( arg[++i] );
            oz = atof( arg[++i] );
            continue;
        }
    }

    if( ( ox == 0. && oy == 0. && oz == 0. ) || radius < 1 )
        error->all( FLERR, "Usage: bounceback/special/meso group [radius double] [center doublex3] [orient doublex3]" );

    double n = std::sqrt( ox * ox + oy * oy + oz * oz );
    ox /= n;
    oy /= n;
    oz /= n;

    nevery = 1;
}

int MesoFixBounceBackSpecial::setmask()
{
    int mask = 0;
    mask |= FixConst::PRE_EXCHANGE;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

void MesoFixBounceBackSpecial::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBounceBackSpecial. %s %d\n", __FILE__, __LINE__ );
    }
}

void MesoFixBounceBackSpecial::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBounceBackSpecial. %s %d\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

// bounce-forward
__global__ void gpu_fix_solid_wall_bounce_special(
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
    int* __restrict Mask,
    const r64 cx,
    const r64 cy,
    const r64 cz,
    const r64 ox,
    const r64 oy,
    const r64 oz,
    const r64 radius,
    const int groupbit,
    const int n_all )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_all; i += gridDim.x * blockDim.x ) {
        if( Mask[i] & groupbit ) {
        	r64 dx = coord_x[i] - cx;
			r64 dy = coord_y[i] - cy;
			r64 dz = coord_z[i] - cz;
			r64 along = dx * ox + dy * oy + dz * oz;
			r64 perpx = dx - along * ox;
			r64 perpy = dy - along * oy;
			r64 perpz = dz - along * oz;
			r64 d = sqrt( perpx*perpx + perpy*perpy + perpz*perpz );
			if( d > radius ) {
				r64 over = d - radius;
				coord_x[i] += over * 2.0 * -perpx/d;
				coord_y[i] += over * 2.0 * -perpy/d;
				coord_z[i] += over * 2.0 * -perpz/d;
				veloc_x[i]  = -veloc_x[i];
				veloc_y[i]  = -veloc_y[i];
				veloc_z[i]  = -veloc_z[i];
			}
        }
    }
}

void MesoFixBounceBackSpecial::bounce_back()
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_solid_wall_bounce_special, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_solid_wall_bounce_special, cudaFuncCachePreferL1 );
    }

    gpu_fix_solid_wall_bounce_special <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
		meso_atom->dev_coord[0], meso_atom->dev_coord[1], meso_atom->dev_coord[2],
		meso_atom->dev_veloc[0], meso_atom->dev_veloc[1], meso_atom->dev_veloc[2],
        meso_atom->dev_mask,
        cx, cy, cz,
        ox, oy, oz,
        radius,
        groupbit,
        atom->nlocal );
}

void MesoFixBounceBackSpecial::end_of_step()
{
    bounce_back();
}

void MesoFixBounceBackSpecial::pre_exchange()
{
    bounce_back();
}

