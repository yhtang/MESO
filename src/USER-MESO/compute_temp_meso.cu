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
#include "compute_temp_meso.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;

MesoComputeTemp::MesoComputeTemp( LAMMPS *lmp, int narg, char **arg ) :
    Compute( lmp, narg, arg ),
    MesoPointers( lmp ),
    t( lmp, "MesoComputeTemp::t" ),
    per_atom_eK( lmp, "MesoComputeTemp::per_atom_Ek" )
{
    if( narg != 3 ) error->all( __FILE__, __LINE__, "Illegal compute temp command" );

    scalar_flag = 1;
    vector_flag = 0;
    extscalar = 0;
    tempflag = 1;
    fix_dof = 0;
    tfactor = 0;

    t.grow( 1 );
    if( atom->nlocal ) per_atom_eK.grow( atom->nlocal );
}

MesoComputeTemp::~MesoComputeTemp()
{
}

void MesoComputeTemp::setup()
{
    fix_dof = 0;
    for( int i = 0; i < modify->nfix; i++ )
        fix_dof += modify->fix[i]->dof( igroup );
    dof_compute();
}

void MesoComputeTemp::dof_compute()
{
    double natoms = group->count( igroup );
    dof = domain->dimension * natoms;
    dof -= extra_dof + fix_dof;
    if( dof > 0.0 ) tfactor = force->mvv2e / ( dof * force->boltz );
    else tfactor = 0.0;
}

__global__ void gpu_eK_scalar(
    r64* __restrict veloc_x,
    r64* __restrict veloc_y,
    r64* __restrict veloc_z,
    r64* __restrict mass,
    int* __restrict mask,
    r64* __restrict eK,
    int  groupbit,
    int  n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if( i >= n ) return;

    if( mask[i] & groupbit ) {
        eK[i] = mass[i] * ( veloc_x[i] * veloc_x[i] + veloc_y[i] * veloc_y[i] + veloc_z[i] * veloc_z[i] );
    }
}

double MesoComputeTemp::compute_scalar()
{
    invoked_scalar = update->ntimestep;

    if( atom->nlocal > per_atom_eK.n_elem() ) per_atom_eK.grow( atom->nlocal, false, false );
    size_t threads_per_block = meso_device->query_block_size( gpu_eK_scalar );
    gpu_eK_scalar <<< n_block( atom->nlocal, threads_per_block ), threads_per_block, 0, meso_device->stream() >>> (
        meso_atom->dev_veloc(0),
        meso_atom->dev_veloc(1),
        meso_atom->dev_veloc(2),
        meso_atom->dev_mass,
        meso_atom->dev_mask,
        per_atom_eK,
        groupbit,
        atom->nlocal );
    threads_per_block = meso_device->query_block_size( gpu_reduce_sum_host<double> ) ;
    gpu_reduce_sum_host <<< 1, threads_per_block, 0, meso_device->stream() >>> (
        per_atom_eK.ptr(), t.ptr(), atom->nlocal );
    meso_device->stream().sync();

    MPI_Allreduce( t, &scalar, 1, MPI_DOUBLE, MPI_SUM, world );
    if( dynamic ) dof_compute();
    scalar *= tfactor;
    return scalar;
}
