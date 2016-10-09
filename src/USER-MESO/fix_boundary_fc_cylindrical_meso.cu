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
#include "fix_boundary_fc_cylindrical_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryFcCylindrical::MesoFixBoundaryFcCylindrical( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    cut( 0. ), a0( 0. ),
    cx( 0. ), cy( 0. ), cz( 0. ),
    ox( 0. ), oy( 0. ), oz( 0. ),
    radius( 0. ), length( 0. ),
    poly( lmp, "MesoFixBoundaryFcCylindrical::poly" )
{
    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "a0" ) ) {
            a0 = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "radius" ) ) {
            radius = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "length" ) ) {
            length = atof( arg[++i] );
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
        if( !strcmp( arg[i], "poly" ) ) {
            int order = atoi( arg[++i] );
            poly.grow( order + 1 );
            for( int j = 0; j < order + 1; j++ ) poly[j] = atof( arg[++i] );
            continue;
        }
    }

    if( ( ox == 0. && oy == 0. && oz == 0. ) || radius < 1 || poly.n_elem() == 0 || poly == NULL )
        error->all( FLERR, "Usage: boundary/fc group [type int] [T0 double] [cut double] [radius double] [length double] [center doublex3] [orient doublex3] [poly int doublex?]" );

    double n = std::sqrt( ox * ox + oy * oy + oz * oz );
    ox /= n;
    oy /= n;
    oz /= n;
}

MesoFixBoundaryFcCylindrical::~MesoFixBoundaryFcCylindrical()
{
}

int MesoFixBoundaryFcCylindrical::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryFcCylindrical::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcCylindrical. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryFcCylindrical::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFcCylindrical. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_fc_cylindrical(
    r64* __restrict coord_x,
    r64* __restrict coord_y,
    r64* __restrict coord_z,
    r64* __restrict force_x,
    r64* __restrict force_y,
    r64* __restrict force_z,
    int* __restrict mask,
    const int groupbit,
    const int order,
    r64* __restrict poly,
    const r64 a0,
    const r64 cx,
    const r64 cy,
    const r64 cz,
    const r64 ox,
    const r64 oy,
    const r64 oz,
    const r64 cut,
    const r64 radius,
    const r64 length,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
        	r64 dx = coord_x[i] - cx;
        	r64 dy = coord_y[i] - cy;
        	r64 dz = coord_z[i] - cz;
        	r64 along = dx * ox + dy * oy + dz * oz;
        	r64 perpx = dx - along * ox;
        	r64 perpy = dy - along * oy;
        	r64 perpz = dz - along * oz;
        	r64 d = sqrt( perpx*perpx + perpy*perpy + perpz*perpz );
            if( d > radius - cut ) {
                r64 h         = max( min( radius - d, cut ), 0. );
                r64 wc        = max( polyval( h, order, poly ), 0. );
                r64 force     = a0 * wc;
                force_x[i]   -= force * perpx / d;
                force_y[i]   -= force * perpy / d;
                force_z[i]   -= force * perpz / d;
            }
        }
    }
}

void MesoFixBoundaryFcCylindrical::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_fc_cylindrical, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_fc_cylindrical, cudaFuncCachePreferL1 );
    }

    gpu_fix_boundary_fc_cylindrical <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        meso_atom->dev_mask,
        groupbit,
        poly.n_elem() - 1,
        poly,
        a0,
        cx, cy, cz,
        ox, oy, oz,
        cut, radius, length,
        atom->nlocal );
}
