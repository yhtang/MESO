#include "stdlib.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "bond.h"
#include "force.h"
#include "atom_vec.h"
#include "pointers.h"
#include "domain.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "domain_meso.h"
#include "error_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

#define DELTA 10000

const int    MesoAtomVec::growth_inc = 65536 ;
const double MesoAtomVec::growth_mul = 1.2   ;

MesoAtomVec::MesoAtomVec( LAMMPS *lmp ) :
    MesoPointers( lmp ),
    // device buffers
    dev_perm_key( lmp, "MesoAtomVec::dev_perm_key" ),
    dev_permute_from( lmp, "MesoAtomVec::dev_permute_from" ),
    dev_permute_to( lmp, "MesoAtomVec::dev_permute_to" ),
    dev_tag( lmp, "MesoAtomVec::dev_tag" ),
    dev_type( lmp, "MesoAtomVec::dev_type" ),
    dev_mask( lmp, "MesoAtomVec::dev_mask" ),
    dev_mass( lmp, "MesoAtomVec::dev_mass" ),
    dev_coord( lmp, "MesoAtomVec::dev_coord", 3 ),
    dev_force( lmp, "MesoAtomVec::dev_force", 3 ),
    dev_veloc( lmp, "MesoAtomVec::dev_veloc", 3 ),
    dev_virial( lmp, "MesoAtomVec::dev_virial", 6 ),
    dev_image( lmp, "MesoAtomVec::dev_image" ),
//  dev_e_dihed         ( lmp, "MesoAtomVec::dev_e_dihed" ),
//  dev_e_impro         ( lmp, "MesoAtomVec::dev_e_impro" ),
    dev_e_pair( lmp, "MesoAtomVec::dev_e_pair" ),
    dev_r_coord( lmp, "MesoAtomVec::dev_r_coord", 3 ),
    dev_r_veloc( lmp, "MesoAtomVec::dev_r_veloc", 3 ),
    dev_coord_merged( lmp, "MesoAtomVec::dev_coord_merged" ),
    dev_veloc_merged( lmp, "MesoAtomVec::dev_veloc_merged" ),
    hst_borderness( lmp, "MesoAtomVec::hst_borderness" ),
    dev_coord_pinned( lmp, "MesoAtomVec::dev_coord_pinned" ),
    dev_veloc_pinned( lmp, "MesoAtomVec::dev_veloc_pinned" ),
    dev_force_pinned( lmp, "MesoAtomVec::dev_force_pinned" ),
    dev_image_pinned( lmp, "MesoAtomVec::dev_image_pinned" ),
    dev_tag_pinned( lmp, "MesoAtomVec::dev_tag_pinned" ),
    dev_type_pinned( lmp, "MesoAtomVec::dev_type_pinned" ),
    dev_mask_pinned( lmp, "MesoAtomVec::dev_mask_pinned" ),
    dev_masstype_pinned( lmp, "MesoAtomVec::dev_masstype_pinned" ),
    devNImprop( lmp, "MesoAtomVec::devNImprop" ),
    devImprops( lmp, "MesoAtomVec::devImprops" ),
    devImpropType( lmp, "MesoAtomVec::devImpropType" )
{
    pre_sort     = AtomAttribute::NONE;
    post_sort    = AtomAttribute::NONE;
    pre_border   = AtomAttribute::NONE;
    post_border  = AtomAttribute::NONE;
    pre_comm     = AtomAttribute::NONE;
    post_comm    = AtomAttribute::NONE;
    pre_exchange = AtomAttribute::NONE;
    pre_output   = AtomAttribute::NONE;
    excl_table_padding = 0;
    alloced = 0;
}

MesoAtomVec::~MesoAtomVec()
{
}

void MesoAtomVec::pin_host_array()
{
    meso_device->sync_device();
    if( meso_atom->nmax ) {
        dev_coord_pinned.map_host( 3 * meso_atom->nmax, &( meso_atom->x[0][0] ) );
        dev_force_pinned.map_host( 3 * meso_atom->nmax, &( meso_atom->f[0][0] ) );
        dev_veloc_pinned.map_host( 3 * meso_atom->nmax, &( meso_atom->v[0][0] ) );
        dev_image_pinned.map_host( meso_atom->nmax, meso_atom->image );
        dev_type_pinned .map_host( meso_atom->nmax, meso_atom->type );
        dev_tag_pinned  .map_host( meso_atom->nmax, meso_atom->tag );
        dev_mask_pinned .map_host( meso_atom->nmax, meso_atom->mask );
    }
    if( meso_atom->ntypes ) dev_masstype_pinned .map_host( meso_atom->ntypes + 1, meso_atom->mass );
}

void MesoAtomVec::unpin_host_array()
{
    meso_device->sync_device();
    dev_coord_pinned   .unmap_host( &( meso_atom->x[0][0] ) );
    dev_force_pinned   .unmap_host( &( meso_atom->f[0][0] ) );
    dev_veloc_pinned   .unmap_host( &( meso_atom->v[0][0] ) );
    dev_image_pinned   .unmap_host( meso_atom->image );
    dev_type_pinned    .unmap_host( meso_atom->type );
    dev_tag_pinned     .unmap_host( meso_atom->tag );
    dev_mask_pinned    .unmap_host( meso_atom->mask );
    dev_masstype_pinned.unmap_host( meso_atom->mass );
}

void MesoAtomVec::grow_device( int nmax_new )
{
#ifdef LMP_MESO_LOG_L3
    fprintf( stderr, "<MESO> meso_atom std::vector grew from %d to %d\n", nmax , nmax_new );
#endif

    meso_device->sync_device();

    // per-meso_atom array
    meso_atom->dev_perm_key      = dev_perm_key     .grow( nmax_new, false, false );
    meso_atom->dev_permute_from = dev_permute_from.grow( nmax_new, false, false );
    meso_atom->dev_permute_to = dev_permute_to.grow( nmax_new, false, false );

    meso_atom->dev_tag          = dev_tag    .grow( nmax_new );
    meso_atom->dev_type         = dev_type   .grow( nmax_new );
    meso_atom->dev_mask         = dev_mask   .grow( nmax_new );
    meso_atom->dev_mass         = dev_mass   .grow( nmax_new );
    meso_atom->dev_coord        = dev_coord  .grow( nmax_new );
    meso_atom->dev_veloc        = dev_veloc  .grow( nmax_new );
    meso_atom->dev_force        = dev_force  .grow( nmax_new );
    meso_atom->dev_virial       = dev_virial .grow( nmax_new );
    meso_atom->dev_image        = dev_image  .grow( nmax_new );
    meso_atom->dev_e_pair       = dev_e_pair .grow( nmax_new, false, true );
    meso_atom->dev_r_coord      = dev_r_coord.grow( nmax_new, false, false );
    meso_atom->dev_r_veloc      = dev_r_veloc.grow( nmax_new, false, false );
    meso_atom->dev_coord_merged = dev_coord_merged.grow( nmax_new, false, false );
    meso_atom->dev_veloc_merged = dev_veloc_merged.grow( nmax_new, false, false );
    meso_atom->hst_borderness   = hst_borderness.grow( nmax_new, false, false );

    // texture mapping
    meso_atom->tex_tag         .bind( dev_tag );
    meso_atom->tex_coord_merged.bind( dev_coord_merged );
    meso_atom->tex_veloc_merged.bind( dev_veloc_merged );

    // topology left to each derived meso_atom vectors
}

__global__ void gpu_merge_xvt(
    r64* __restrict coord_x, r64* __restrict coord_y, r64* __restrict coord_z,
    r64* __restrict veloc_x, r64* __restrict veloc_y, r64* __restrict veloc_z,
    int* __restrict type, int* __restrict tag,
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
        veloc.w = __uint_as_float( seed ^ premix_TEA<16>( __brev( tag[i] ), __mantissa( veloc.x, veloc.y, veloc.z ) ) );
        veloc_merged[i] = veloc;
    }
}

void MesoAtomVec::dp2sp_merged( int seed, int p_beg, int p_end, bool offset )
{
    r64 cx = 0., cy = 0., cz = 0.;
    if( offset ) {
        cx = 0.5 * ( meso_domain->subhi[0] + meso_domain->sublo[0] );
        cy = 0.5 * ( meso_domain->subhi[1] + meso_domain->sublo[1] );
        cz = 0.5 * ( meso_domain->subhi[2] + meso_domain->sublo[2] );
    }

    static GridConfig grid_cfg;
    if( !grid_cfg.x ) {
        grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_merge_xvt, 0, cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_merge_xvt, cudaFuncCachePreferL1 );
    }

    gpu_merge_xvt <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> (
        dev_coord(0), dev_coord(1), dev_coord(2),
        dev_veloc(0), dev_veloc(1), dev_veloc(2),
        dev_type, dev_tag,
        dev_coord_merged, dev_veloc_merged,
        cx, cy, cz,
        seed,
        p_beg, p_end );
}

int MesoAtomVec::resolve_work_range( AtomAttribute::Descriptor per_atom_prop, int& p_beg, int& p_end )
{
    p_beg = INT_MAX;
    p_end = 0;

    if( ( per_atom_prop & AtomAttribute::LOCAL ) == AtomAttribute::LOCAL ) { // both 2 bits must match
        p_beg = min( p_beg, 0 );
        p_end = max( p_end, meso_atom->nlocal );
    } else {
        if( per_atom_prop & AtomAttribute::BULK ) {
            p_beg = min( p_beg, 0 );
            p_end = max( p_end, meso_atom->n_bulk );
        }
        if( per_atom_prop & AtomAttribute::BORDER ) {
            p_beg = min( p_beg, meso_atom->n_bulk );
            p_end = max( p_end, meso_atom->n_bulk + meso_atom->n_border );
        }
    }
    if( per_atom_prop & AtomAttribute::GHOST ) {
        p_beg = min( p_beg, meso_atom->nlocal );
        p_end = max( p_end, meso_atom->nlocal + meso_atom->nghost );
    }

    return p_end - p_beg;
}

std::vector<CUDAEvent> MesoAtomVec::transfer( AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int* permute_from, int* permute_to, int action, bool trainable )
{
    const int p_stream_init = 1; // use different streams for data transfer other than the main
    int p_stream = p_stream_init, p_beg, p_end, n_atom;
    std::vector<CUDAEvent> events;

    n_atom = resolve_work_range( per_atom_prop, p_beg, p_end );
    if( !n_atom ) {
        if( meso_domain->periodicity[0] || meso_domain->periodicity[1] || meso_domain->periodicity[2] ) {
            char info[512];
            sprintf( info, "[CDEV] nothing to transfer (%s %s %s).\n",
                     ( per_atom_prop & AtomAttribute::BULK ) ? ( "bulk" ) : ( "" ),
                     ( per_atom_prop & AtomAttribute::BORDER ) ? ( "border" ) : ( "" ),
                     ( per_atom_prop & AtomAttribute::GHOST ) ? ( "ghost" ) : ( "" )
                   );
            meso_error->warning( FLERR, info );
        }
        return events;
    }

    static ThreadTuner &o = meso_device->tuner( "MesoAtomVec::transfer", 1, 2 );
    bool streamed = ( o.bet() == 1 ? false : true );
    int p_inc = streamed ? 1 : 0;

    // transfer
    transfer_impl( events, per_atom_prop, direction, p_beg, n_atom, p_stream,
                   p_inc, permute_to, permute_from, action, streamed );

    if( !streamed ) {
        CUDAEvent e = meso_device->event( "MesoAtomVec::transfer" );
        e.record( meso_device->stream( p_stream_init ) );
        events.clear();
        events.push_back( e );
    }

    return events;
}

void MesoAtomVec::transfer_impl(
    std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction,
    int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed )
{
    if( per_atom_prop & AtomAttribute::COORD ) {
        events.push_back(
            transfer_vector<3, r64, 128>( dev_coord_pinned, dev_coord,
                                          direction, permute_to, p_beg, n_atom,
                                          meso_device->stream( p_stream += p_inc ), action,
                                          streamed ) );
    }
    if( per_atom_prop & AtomAttribute::FORCE ) {
        events.push_back(
            transfer_vector<3, r64, 128>( dev_force_pinned, dev_force,
                                          direction, permute_to, p_beg, n_atom,
                                          meso_device->stream( p_stream += p_inc ), action,
                                          streamed ) );
    }
    if( per_atom_prop & AtomAttribute::VELOC ) {
        events.push_back(
            transfer_vector<3, r64, 128>( dev_veloc_pinned, dev_veloc,
                                          direction, permute_to, p_beg, n_atom,
                                          meso_device->stream( p_stream += p_inc ), action,
                                          streamed ) );
    }
    if( per_atom_prop & AtomAttribute::IMAGE ) {
        events.push_back(
            transfer_scalar( dev_image_pinned, dev_image, direction,
                             permute_from, p_beg, n_atom,
                             meso_device->stream( p_stream += p_inc ), action,
                             streamed ) );
    }
    if( per_atom_prop & AtomAttribute::TYPE ) {
        int p_stream_for_type = p_stream;
        events.push_back(
            transfer_scalar( dev_type_pinned, dev_type, direction,
                             permute_from, p_beg, n_atom,
                             meso_device->stream( p_stream += p_inc ), action,
                             streamed ) );
        if( per_atom_prop & AtomAttribute::MASS ) {
            events.push_back(
                unpack_by_type( dev_masstype_pinned, dev_type, dev_mass,
                                meso_atom->ntypes + 1, direction, p_beg, n_atom,
                                meso_device->stream( p_stream_for_type ), action,
                                streamed ) );
        }
    }
    if( ( per_atom_prop & AtomAttribute::MASS ) && !( per_atom_prop & AtomAttribute::TYPE ) ) {
        meso_error->all( FLERR,
                         "<MESO> cannot unpack mass without transfering particle type." );
    }
    if( per_atom_prop & AtomAttribute::TAG ) {
        events.push_back(
            transfer_scalar( dev_tag_pinned, dev_tag, direction,
                             permute_from, p_beg, n_atom,
                             meso_device->stream( p_stream += p_inc ), action,
                             streamed ) );
    }
    if( per_atom_prop & AtomAttribute::MASK ) {
        events.push_back(
            transfer_scalar( dev_mask_pinned, dev_mask, direction,
                             permute_from, p_beg, n_atom,
                             meso_device->stream( p_stream += p_inc ), action,
                             streamed ) );
    }
}

void MesoAtomVec::force_clear( AtomAttribute::Descriptor range, int vflag )
{
    // clear force on all particles
    // newton flag is always off in MESO-MVV, so never include ghosts
    int p_beg, p_end, n_work;
    resolve_work_range( range, p_beg, p_end );
    if( meso_neighbor->includegroup ) p_end = min( p_end, meso_atom->nfirst );
    n_work = p_end - p_beg;

    dev_force.set( 0.0, meso_device->stream(), p_beg, n_work );
    if( vflag ) dev_virial.set( 0.0, meso_device->stream(), p_beg, n_work );
}

