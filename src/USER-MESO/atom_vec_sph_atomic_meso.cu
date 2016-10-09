#include "stdlib.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"

#include "meso.h"
#include "atom_meso.h"
#include "domain_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "atom_vec_sph_atomic_meso.h"

using namespace LAMMPS_NS;

void AtomVecSPHAtomic::grow( int n )
{
    unpin_host_array();
    if( n == 0 ) n = max( nmax + growth_inc, ( int )( nmax * growth_mul ) );
    grow_cpu( n );
    grow_device( n );
    pin_host_array();
}

void AtomVecSPHAtomic::grow_cpu( int n )
{
    AtomVecSPH::grow( n );
}

void AtomVecSPHAtomic::grow_device( int nmax_new )
{
    MesoAtomVec::grow_device( nmax_new );

    meso_atom->dev_rho = dev_rho.grow( nmax_new, false, true );

    meso_atom->tex_rho .bind( dev_rho );
    meso_atom->tex_mass.bind( dev_mass );
}

void AtomVecSPHAtomic::pin_host_array()
{
    MesoAtomVec::pin_host_array();

    if( atom->rho ) dev_rho_pinned.map_host( atom->nmax, atom->rho );
}

void AtomVecSPHAtomic::unpin_host_array()
{
    MesoAtomVec::unpin_host_array();

    dev_rho_pinned.unmap_host( atom->rho );
}

void AtomVecSPHAtomic::data_atom_target( int i, double *coord, int imagetmp, char **values )
{
    error->one( __FILE__, __LINE__, "[MESO] undefined function." );
}

__global__ void gpu_merge_xvt_sph(
    r64* __restrict coord_x, r64* __restrict coord_y, r64* __restrict coord_z,
    r64* __restrict veloc_x, r64* __restrict veloc_y, r64* __restrict veloc_z,
    r64* __restrict mass, int* __restrict type,
    float4* __restrict coord_merged, float4* __restrict veloc_merged,
    const r64 cx, const r64 cy, const r64 cz,
    const int seed,
    const int p_beg,
    const int p_end )
{
    for( int i  = p_beg + blockDim.x * blockIdx.x + threadIdx.x; i < p_end; i += gridDim.x * blockDim.x ) {
        float4 coord;
        coord.x = coord_x[i] - cx;
        coord.y = coord_y[i] - cy;
        coord.z = coord_z[i] - cz;
        coord.w = __int_as_float( type[i] - 1 );
        coord_merged[i] = coord;

        float4 veloc;
        veloc.x = veloc_x[i];
        veloc.y = veloc_y[i];
        veloc.z = veloc_z[i];
        veloc.w = mass[i];
        veloc_merged[i] = veloc;
    }
}

void AtomVecSPHAtomic::dp2sp_merged( int seed, int p_beg, int p_end, bool offset )
{
    r64 cx = 0., cy = 0., cz = 0.;
    if( offset ) {
        cx = 0.5 * ( meso_domain->subhi[0] + meso_domain->sublo[0] );
        cy = 0.5 * ( meso_domain->subhi[1] + meso_domain->sublo[1] );
        cz = 0.5 * ( meso_domain->subhi[2] + meso_domain->sublo[2] );
    }

    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_merge_xvt_sph, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_merge_xvt_sph, cudaFuncCachePreferL1 );
    }

    gpu_merge_xvt_sph <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        meso_atom->dev_coord(0), meso_atom->dev_coord(1), meso_atom->dev_coord(2),
        meso_atom->dev_veloc(0), meso_atom->dev_veloc(1), meso_atom->dev_veloc(2),
        meso_atom->dev_mass, meso_atom->dev_type,
        meso_atom->dev_coord_merged, meso_atom->dev_veloc_merged,
        cx, cy, cz,
        seed,
        p_beg, p_end );
}

void AtomVecSPHAtomic::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    MesoAtomVec::transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream, p_inc, permute_to, permute_from, action, streamed );
    p_stream = events.size() + p_inc;

    if( per_atom_prop & AtomAttribute::RHO ) {
        events.push_back(
            transfer_scalar(
                dev_rho_pinned, dev_rho, direction, permute_from, p_beg, n_atom, meso_device->stream( p_stream += p_inc ), action ) );
    }
}

void AtomVecSPHAtomic::force_clear( AtomAttribute::Descriptor range, int vflag )
{
    // clear force on all particles
    // newton flag is always off in MESO-MVV, so never include ghosts
    int p_beg, p_end, n_work;
    resolve_work_range( range, p_beg, p_end );
    if( meso_neighbor->includegroup ) p_end = min( p_end, meso_atom->nfirst );
    n_work = p_end - p_beg;

    dev_force.set( 0.0, meso_device->stream(), p_beg, n_work );
    dev_rho.set( 0.0, meso_device->stream(), p_beg, n_work );
}

