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
#include "fix_boundary_fc_meso.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MesoFixBoundaryFc::MesoFixBoundaryFc( LAMMPS *lmp, int narg, char **arg ):
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    cut( 0. ),
    H( 0. ),
    nx( 0. ), ny( 0. ), nz( 0. ),
    poly( lmp, "MesoFixBoundaryFc::poly" )
{
    bool set_H_by_example_point = false;
    double px, py, pz;

    for( int i = 3; i < narg; i++ ) {
        if( !strcmp( arg[i], "cut" ) ) {
            cut = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "H" ) ) {
            H = atof( arg[++i] );
            continue;
        }
        if( !strcmp( arg[i], "p" ) ) {
            px = atof( arg[++i] );
            py = atof( arg[++i] );
            pz = atof( arg[++i] );
            set_H_by_example_point = true;
            continue;
        }
        if( !strcmp( arg[i], "n" ) ) {
            nx = atof( arg[++i] );
            ny = atof( arg[++i] );
            nz = atof( arg[++i] );
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
        error->all( FLERR, "Usage: boundary/fc group [cut double] [H double]|[p doublex3] [n doublex3] [poly int doublex?]" );

    double n = std::sqrt( nx * nx + ny * ny + nz * nz );
    nx /= n;
    ny /= n;
    nz /= n;

    if( set_H_by_example_point ) {
        H = px * nx + py * ny + pz * nz;
    }
}

MesoFixBoundaryFc::~MesoFixBoundaryFc()
{
}

int MesoFixBoundaryFc::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_FORCE;
    return mask;
}

void MesoFixBoundaryFc::init()
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFc. %s %cut\n", __FILE__, __LINE__ );
    }
}

void MesoFixBoundaryFc::setup( int vflag )
{
    if( strcmp( update->integrate_style, "respa" ) == 0 ) {
        fprintf( stderr, "<MESO> RESPA not supported in MesoFixBoundaryFc. %s %cut\n", __FILE__, __LINE__ );
    }
    post_force( vflag );
}

__global__ void gpu_fix_boundary_fc(
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
    const r64 nx,
    const r64 ny,
    const r64 nz,
    const r64 cut,
    const r64 H,
    const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 d = coord_x[i] * nx + coord_y[i] * ny + coord_z[i] * nz;
            if( d > H - cut ) {
                r64 h       = max( min( H - d, cut ), 0. );
                r64 force   = max( polyval( h, order, poly ), 0. );
                force_x[i] += force * -nx;
                force_y[i] += force * -ny;
                force_z[i] += force * -nz;
            }
        }
    }
}

void MesoFixBoundaryFc::post_force( int vflag )
{
    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fix_boundary_fc, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_fix_boundary_fc, cudaFuncCachePreferL1 );
    }

    gpu_fix_boundary_fc <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
        meso_atom->dev_mask,
        groupbit,
        poly.n_elem() - 1,
        poly,
        nx, ny, nz,
        cut, H,
        atom->nlocal );
}
