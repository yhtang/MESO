#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "atom_vec.h"
#include "update.h"
#include "force.h"
#include "group.h"
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
#include "fix_density_profile_spherical_meso.h"

using namespace LAMMPS_NS;

static const r64 PI = 3.14159265358979323846;

MesoFixDensitySpherical::MesoFixDensitySpherical( LAMMPS *lmp, int narg, char **arg ) :
    Fix( lmp, narg, arg ),
    MesoPointers( lmp ),
    dev_com( lmp, "MesoFixDensitySpherical::dev_com" ),
    dev_polar_grids1( lmp, "MesoFixDensitySpherical::dev_polar_grids1" ),
    dev_polar_grids2( lmp, "MesoFixDensitySpherical::dev_polar_grids2" ),
    dev_polar_grid_size1( lmp, "MesoFixDensitySpherical::dev_polar_grid_size1" ),
    dev_polar_grid_size2( lmp, "MesoFixDensitySpherical::dev_polar_grid_size2" ),
    dev_polar_grid_meanr1( lmp, "MesoFixDensitySpherical::dev_polar_grid_meanr1" ),
    dev_polar_grid_meanr2( lmp, "MesoFixDensitySpherical::dev_polar_grid_meanr2" ),
    dev_density_profile( lmp, "MesoFixDensitySpherical::dev_density_profile" )
{
    if( narg < 8 ) error->all( __FILE__, __LINE__, "<MESO> density/spherical/meso usage: id membrane_group style target_group filename grid_size rmax dr [beg end]" );

    int jgroup;
    if( ( jgroup = group->find( arg[3] ) ) == -1 ) {
        error->all( FLERR, "<MESO> Undefined group id in density/spherical/meso usage" );
    }
    target_groupbit = group->bitmask[ jgroup ];
    int n_particle = 0;
    for( int i = 0; i < atom->nlocal; i++ ) if( atom->mask[i] & groupbit ) n_particle++;

	filename  = "";
    every     =  0;
	window    =  0;
	da        =  0;
	maxr      =  0;
	dr        =  0;
	n_sample  =  0;

	for( int i = 0 ; i < narg ; i++ )
	{
		if ( !strcmp( arg[i], "output" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'output'");
			filename = arg[i];
		}
		else if ( !strcmp( arg[i], "da" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'nbin'");
			da = atof( arg[i] );
		}
		else if ( !strcmp( arg[i], "maxr" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'nbin'");
			maxr = atof( arg[i] );
		}
		else if ( !strcmp( arg[i], "dr" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'nbin'");
			dr = atof( arg[i] );
		}
		else if ( !strcmp( arg[i], "every" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'every'");
			every = atoi( arg[i] );
		}
		else if ( !strcmp( arg[i], "window" ) )
		{
			if ( ++i >= narg ) error->all(FLERR,"Incomplete density/spherical/meso command after 'window'");
			window = atoi( arg[i] );
		}
	}

	if ( filename == "" || every == 0 || window == 0 || da == 0 || dr == 0 || maxr == 0 ) {
		error->all(FLERR,"Incomplete density/spherical/meso command: insufficient arguments");
	}
	if ( window * 2 >= every ) {
		error->warning(FLERR,"density/spherical/meso: window size larger than sampling length");
	}

    n_grid_t = std::ceil( 1. / da );
    n_grid_p = std::ceil( 2. / da );
    n_grid = n_grid_t * n_grid_p;
    int n_particle_per_grid = n_particle / n_grid;
    dev_polar_grid_size1.grow( n_grid );
    dev_polar_grid_size2.grow( n_grid );
    dev_polar_grid_meanr1.grow( n_grid );
    dev_polar_grid_meanr2.grow( n_grid );
    dev_polar_grids1.grow( n_grid, n_particle_per_grid * 4 );
    dev_polar_grids2.grow( n_grid, n_particle_per_grid * 4 );
    dev_com.grow( 4 );
    n_half_bin = std::ceil( maxr / dr );
    dev_density_profile.grow( 2 * n_half_bin + 1 );
    dev_density_profile.set( 0, meso_device->stream() );
}

int MesoFixDensitySpherical::setmask()
{
    int mask = 0;
    mask |= FixConst::POST_INTEGRATE;
    return mask;
}

void MesoFixDensitySpherical::setup( int vflag )
{
    post_integrate();
}

void MesoFixDensitySpherical::dump( bigint tstamp ) {
	if ( n_sample ) {
		std::ofstream fout;
		char fn[256];
		sprintf( fn, "%s.%09d", filename.c_str(), tstamp );
		fout.open( fn );

		std::vector<r64> profile(dev_density_profile.n_elem(), 0);
		dev_density_profile.download(profile.data(), profile.size());
		for (int i = 0; i < profile.size(); i++) {
			fout << i * dr - maxr << ( (i != profile.size() - 1) ? '\t' : '\n' );
		}
		for (int i = 0; i < profile.size(); i++) {
			fout << profile[i] / n_sample << ( (i != profile.size() - 1) ? '\t' : '\n' );
		}
		fout.close();
		n_sample = 0;
		dev_density_profile.set( 0, meso_device->stream() );
	}
}

MesoFixDensitySpherical::~MesoFixDensitySpherical()
{
	bigint step = update->ntimestep;
	bigint n = ( step + every/2 ) / every;
	bigint m = n * every;
	if ( step - m >= -window && step - m < window ) {
		if ( last_dump_time < step ) {
			dump( m );
		}
	}
}

__global__ void gpu_compute_com(
    r64 * __restrict coord_x,
    r64 * __restrict coord_y,
    r64 * __restrict coord_z,
    int * __restrict mask,
    r64 * __restrict mass,
    r32 * __restrict com,
    const int groupbit,
    const int n
)
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            atomic_add( com + 0, coord_x[i] * mass[i] );
            atomic_add( com + 1, coord_y[i] * mass[i] );
            atomic_add( com + 2, coord_z[i] * mass[i] );
            atomic_add( com + 3, mass[i] );
        }
    }
}

__global__ void gpu_grid_spherical(
    r64 * __restrict coord_x,
    r64 * __restrict coord_y,
    r64 * __restrict coord_z,
    int * __restrict mask,
    r32 * __restrict com,
    int * __restrict polar_grids1,
    int * __restrict polar_grids2,
    uint* __restrict polar_grid_size1,
    uint* __restrict polar_grid_size2,
    const int polar_grids_padding,
    const r64 da,
    const int groupbit,
    const int n_grid_t,
    const int n_grid_p,
    const int max_grid_sz,
    const int n
)
{
    r64 cx = com[0] / com[3];
    r64 cy = com[1] / com[3];
    r64 cz = com[2] / com[3];
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 x = coord_x[i] - cx;
            r64 y = coord_y[i] - cy;
            r64 z = coord_z[i] - cz;
            r64 r = sqrt( x * x + y * y + z * z );
            // grid1
            r64 theta = acos( z / r );
            r64 phi = atan2( y, x );
            int bid_t = clamp( theta   / PI / da, 0, n_grid_t );
            int bid_p = clamp( ( phi + PI ) / PI / da, 0, n_grid_p );
            int bid = bid_t * n_grid_p + bid_p;
            int p = atomicInc( polar_grid_size1 + bid, max_grid_sz );
            polar_grids1[ bid + p * polar_grids_padding ] = i;
            // grid2
            theta = acos( y / r );
            phi = atan2( z, x );
            bid_t = clamp( theta   / PI / da, 0, n_grid_t );
            bid_p = clamp( ( phi + PI ) / PI / da, 0, n_grid_p );
            bid = bid_t * n_grid_p + bid_p;
            p = atomicInc( polar_grid_size2 + bid, max_grid_sz );
            polar_grids2[ bid + p * polar_grids_padding ] = i;
        }
    }
}

__global__ void check( uint * polar_grid_size, const int n )
{
    for( int i = 0; i < n; i++ ) printf( "grid %d np %d\n", i, polar_grid_size[i] );
}

__global__ void gpu_measure_meanr(
    r64 * __restrict coord_x,
    r64 * __restrict coord_y,
    r64 * __restrict coord_z,
    r32 * __restrict com,
    int * __restrict polar_grids,
    uint* __restrict polar_grid_size,
    r64 * __restrict polar_grid_meanr,
    const int polar_grids_padding,
    const int n_grid )
{
    r64 cx = com[0] / com[3];
    r64 cy = com[1] / com[3];
    r64 cz = com[2] / com[3];
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_grid; i += gridDim.x * blockDim.x ) {
        r64 meanr = 0.;
        for( int j = 0, *pgrid = polar_grids + i; j < polar_grid_size[i]; j++, pgrid += polar_grids_padding ) {
            int k = *pgrid;
            r64 x = coord_x[k] - cx;
            r64 y = coord_y[k] - cy;
            r64 z = coord_z[k] - cz;
            r64 r = sqrt( x * x + y * y + z * z );
            meanr += r;
        }
        polar_grid_meanr[i] = meanr / polar_grid_size[i];
    }
}

__global__ void gpu_density_profile(
    r64 * __restrict coord_x,
    r64 * __restrict coord_y,
    r64 * __restrict coord_z,
    int * __restrict  mask,
    r32 * __restrict  com,
    uint* __restrict polar_grid_size1,
    uint* __restrict polar_grid_size2,
    r64 * __restrict  polar_grid_meanr1,
    r64 * __restrict  polar_grid_meanr2,
    r64 * __restrict  density_profile,
    const int groupbit,
    const r64 da,
    const r64 maxr,
    const r64 dr,
    const int n_grid_t,
    const int n_grid_p,
    const int n_half_bin,
    const int n )
{
    r64 cx = com[0] / com[3];
    r64 cy = com[1] / com[3];
    r64 cz = com[2] / com[3];
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
        if( mask[i] & groupbit ) {
            r64 x = coord_x[i] - cx;
            r64 y = coord_y[i] - cy;
            r64 z = coord_z[i] - cz;
            r64 r = sqrt( x * x + y * y + z * z );
            // grid1
            r64 theta = acos( z / r );
            r64 phi = atan2( y, x );
            int bid_t = clamp( theta   / PI / da, 0, n_grid_t );
            int bid_p = clamp( ( phi + PI ) / PI / da, 0, n_grid_p );
            int bid1 = bid_t * n_grid_p + bid_p;
            // grid2
            theta = acos( y / r );
            phi = atan2( z, x );
            bid_t = clamp( theta   / PI / da, 0, n_grid_t );
            bid_p = clamp( ( phi + PI ) / PI / da, 0, n_grid_p );
            int bid2 = bid_t * n_grid_p + bid_p;
            r64 meanr = polar_grid_size1[bid1] > polar_grid_size2[bid2] ? polar_grid_meanr1[bid1] : polar_grid_meanr2[bid2];
            int p = rint( ( r - meanr ) / dr );
            r64 weight = 1.0 / ( 4 * PI * r * r  * dr );
            if( p < n_half_bin && p > -n_half_bin ) atomic_add( density_profile + p + n_half_bin, weight );
        }

    }
}

void MesoFixDensitySpherical::post_integrate()
{
    static GridConfig grid_cfg1, grid_cfg2, grid_cfg3, grid_cfg4;
    if( !grid_cfg1.x ) {
        grid_cfg1 = meso_device->configure_kernel( gpu_compute_com );
        grid_cfg2 = meso_device->configure_kernel( gpu_grid_spherical );
        grid_cfg3 = meso_device->configure_kernel( gpu_measure_meanr );
        grid_cfg4 = meso_device->configure_kernel( gpu_density_profile );
    }

	bigint step = update->ntimestep;
	bigint n = ( step + every/2 ) / every;
	bigint m = n * every;
	if ( step - m >= -window && step - m < window )
	{
        n_sample++;

        dev_com.set( 0.f, meso_device->stream() );
        gpu_compute_com <<< grid_cfg1.x, grid_cfg1.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_mask,
            meso_atom->dev_mass,
            dev_com,
            groupbit,
            atom->nlocal );

        dev_polar_grid_size1.set( 0, meso_device->stream() );
        dev_polar_grid_size2.set( 0, meso_device->stream() );
        gpu_grid_spherical <<< grid_cfg2.x, grid_cfg2.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_mask,
            dev_com,
            dev_polar_grids1,
            dev_polar_grids2,
            dev_polar_grid_size1,
            dev_polar_grid_size2,
            dev_polar_grids1.pitch_elem(),
            da,
            groupbit,
            n_grid_t,
            n_grid_p,
            dev_polar_grids1.h(),
            atom->nlocal );

        gpu_measure_meanr <<< grid_cfg3.x, grid_cfg3.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            dev_com,
            dev_polar_grids1,
            dev_polar_grid_size1,
            dev_polar_grid_meanr1,
            dev_polar_grids1.pitch_elem(),
            n_grid );
        gpu_measure_meanr <<< grid_cfg3.x, grid_cfg3.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            dev_com,
            dev_polar_grids2,
            dev_polar_grid_size2,
            dev_polar_grid_meanr2,
            dev_polar_grids2.pitch_elem(),
            n_grid );

        gpu_density_profile <<< grid_cfg4.x, grid_cfg4.y, 0, meso_device->stream() >>> (
            meso_atom->dev_coord(0),
            meso_atom->dev_coord(1),
            meso_atom->dev_coord(2),
            meso_atom->dev_mask,
            dev_com,
            dev_polar_grid_size1,
            dev_polar_grid_size2,
            dev_polar_grid_meanr1,
            dev_polar_grid_meanr2,
            dev_density_profile,
            target_groupbit,
            da,
            maxr,
            dr,
            n_grid_t,
            n_grid_p,
            n_half_bin,
            atom->nlocal );
    } else if ( step - m == window ) {
		dump( m );
	}

}
