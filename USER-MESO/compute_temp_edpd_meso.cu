#include "mpi.h"
#include "string.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "group.h"
#include "error.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "compute_temp_edpd_meso.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;

MesoComputeTempEDPD::MesoComputeTempEDPD( LAMMPS *lmp, int narg, char **arg ) :
    Compute( lmp, narg, arg ),
    MesoPointers( lmp ),
    t( lmp, "MesoComputeTempEDPD::t" ),
	c( lmp, "MesoComputeTempEDPD::c" )
{
    if( narg != 3 ) error->all( __FILE__, __LINE__, "Illegal compute temp command" );

    scalar_flag = 1;
    vector_flag = 0;
    extscalar = 0;

    t.grow( 1 );
    c.grow( 1 );
}

MesoComputeTempEDPD::~MesoComputeTempEDPD()
{
}

void MesoComputeTempEDPD::setup()
{
}

__global__ void sum_internal_T(
	r64 * __restrict T,
	int * __restrict mask,
	r64 * __restrict sum,
	int * __restrict count,
	const int groupbit,
	const int n
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	r64 t = 0;
	int c = 0;
	if ( i < n ) {
		if ( mask[i] & groupbit ) {
			t = T[i];
			c = 1;
		}
	}
	r64 tsum = __warp_sum( t );
	int csum = __warp_sum( c );
	if ( __laneid() == 0 ) {
		atomic_add( sum, tsum );
		atomic_add( count, csum );
	}
}

double MesoComputeTempEDPD::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    t.set( 0, meso_device->stream() );
    c.set( 0, meso_device->stream() );

    size_t threads_per_block = meso_device->query_block_size( gpu_reduce_sum_host<double> );
    sum_internal_T <<< ( atom->nlocal + threads_per_block - 1 ) / threads_per_block, threads_per_block, 0, meso_device->stream() >>> (
    	meso_atom->dev_T,
    	meso_atom->dev_mask,
    	t,
    	c,
    	groupbit,
        atom->nlocal );

    double tsum;
    int    csum;
    t.download( &tsum, 1, meso_device->stream() );
    c.download( &csum, 1, meso_device->stream() );

    meso_device->stream().sync();

    u64 csum_u64 = csum, total_atoms;
    MPI_Allreduce( &tsum, &scalar, 1, MPI_DOUBLE, MPI_SUM, world );
    MPI_Allreduce( &csum_u64, &total_atoms, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, world );
    scalar /= total_atoms;
    return scalar;
}

