#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"

#include "engine_meso.h"
#include "comm_meso.h"
#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "angle_harmonic_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

#define SMALL 0.000001

MesoAngleHarmonic::MesoAngleHarmonic( LAMMPS *lmp ):
    AngleHarmonic( lmp ),
    MesoPointers( lmp ),
    dev_k( lmp, "MesoAngleHarmonic::dev_k" ),
    dev_theta0( lmp, "MesoAngleHarmonic::dev_theta0" )
{
    coeff_alloced = 0;
}

MesoAngleHarmonic::~MesoAngleHarmonic()
{
}

void MesoAngleHarmonic::alloc_coeff()
{
    if( coeff_alloced ) return;

    coeff_alloced = 1;
    int n = atom->nangletypes;
    dev_k     .grow( n + 1, false, false );
    dev_theta0.grow( n + 1, false, false );
    dev_k     .upload( k,      n + 1, meso_device->stream() );
    dev_theta0.upload( theta0, n + 1, meso_device->stream() );
}

template<int evflag>
__global__ void gpu_angle_harmonic(
    texobj tex_coord_merged,
    r64*  __restrict force_x,
    r64*  __restrict force_y,
    r64*  __restrict force_z,
    r64*  __restrict virial_xx,
    r64*  __restrict virial_yy,
    r64*  __restrict virial_zz,
    r64*  __restrict virial_xy,
    r64*  __restrict virial_xz,
    r64*  __restrict virial_yz,
    r64*  __restrict e_angle,
    int*  __restrict nangle,
    int4* __restrict angle,
    r64*  __restrict k_global,
    r64*  __restrict theta0_global,
    const double3 period,
    const int padding,
    const int n_type,
    const int n_local )
{
    extern __shared__ r64 shared_data[];
    r64 *k      = &shared_data[0];
    r64 *theta0 = &shared_data[n_type + 1];
    for( int i = threadIdx.x ; i < n_type + 1 ; i += blockDim.x ) {
        k     [i] = k_global [i];
        theta0[i] = theta0_global[i];
    }
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nangle[i];

        r64 fx = 0.0, fy = 0.0, fz = 0.0;
        r64 vrxx = 0.0, vryy = 0.0, vrzz = 0.0, vrxy = 0.0, vrxz = 0.0, vryz = 0.0;
        r64 eangle = 0.0 ;

        for( int p = 0 ; p < n ; p++ ) {
            int4 agl = angle[ i + p * padding ];
            int i1 = agl.x;
            int i2 = agl.y;
            int i3 = agl.z;
            int type = agl.w;

            f3i coord1 = tex1Dfetch<float4>( tex_coord_merged, i1 );
            f3i coord2 = tex1Dfetch<float4>( tex_coord_merged, i2 );
            f3i coord3 = tex1Dfetch<float4>( tex_coord_merged, i3 );

            // 1st bond
            r64 delx1 = minimum_image( coord1.x - coord2.x, period.x ) ;
            r64 dely1 = minimum_image( coord1.y - coord2.y, period.y ) ;
            r64 delz1 = minimum_image( coord1.z - coord2.z, period.z ) ;
            r64 rsq1  = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
            r64 rinv1 = rsqrt( rsq1 ) ;

            // 2nd bond
            r64 delx2 = minimum_image( coord3.x - coord2.x, period.x ) ;
            r64 dely2 = minimum_image( coord3.y - coord2.y, period.y ) ;
            r64 delz2 = minimum_image( coord3.z - coord2.z, period.z ) ;
            r64 rsq2  = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
            r64 rinv2 = rsqrt( rsq2 ) ;

            // angle (cos and sin)
            r64 c = delx1 * delx2 + dely1 * dely2 + delz1 * delz2;
            c *= rinv1 * rinv2;
            c = bound( c, -1.0, 1.0 );
            r64 s = rsqrt( max( 1.0 - c * c, SMALL ) );

            // force & energy

            r64 dtheta = acos( c ) - theta0[type];
            r64 tk = k[type] * dtheta;

            r64 a = -2.0 * tk * s;
            r64 a11 =  a * c * rinv1 * rinv1;
            r64 a12 = -a   * rinv1 * rinv2;
            r64 a22 =  a * c * rinv2 * rinv2;

            // apply force to each of 3 atoms

            r64 dfx = 0., dfy = 0., dfz = 0.;
            if( i != i3 ) {
                dfx += a11 * delx1 + a12 * delx2;
                dfy += a11 * dely1 + a12 * dely2;
                dfz += a11 * delz1 + a12 * delz2;
            }
            if( i != i1 ) {
                dfx += a22 * delx2 + a12 * delx1;
                dfy += a22 * dely2 + a12 * dely1;
                dfz += a22 * delz2 + a12 * delz1;
            }

//            if ( dfx != dfx ) {
//            	printf("%d %d %d\n",i1,i2,i3);
//            	printf("%lf %lf %lf %lf %lf %lf\n",delx1,dely1,delz1,delx2,dely2,delz2);
//            }

            fx += ( i != i2 ) ? dfx : -dfx;
            fy += ( i != i2 ) ? dfy : -dfy;
            fz += ( i != i2 ) ? dfz : -dfz;
            if( evflag ) {
                eangle = tk * dtheta;
                f3i coord = ( i == i1 ) ? coord1 : ( ( i == i2 ) ? coord2 : coord3 );
                vrxx += dfx * coord.x;
                vryy += dfy * coord.y;
                vrzz += dfz * coord.z;
                vrxy += dfy * coord.x;
                vrxz += dfz * coord.x;
                vryz += dfz * coord.y;
            }
        }

        force_x[i]   += fx;
        force_y[i]   += fy;
        force_z[i]   += fz;
        if( evflag ) e_angle[i] = eangle * 0.5;
        if( evflag ) {
            virial_xx[i] += vrxx;
            virial_yy[i] += vryy;
            virial_zz[i] += vrzz;
            virial_xy[i] += vrxy;
            virial_xz[i] += vrxz;
            virial_yz[i] += vryz;
            e_angle[i]    = eangle * 0.5;
        }
    }
}

void MesoAngleHarmonic::compute( int eflag, int vflag )
{
	if( !coeff_alloced ) alloc_coeff();

    static GridConfig grid_cfg, grid_cfg_EV;
    if( !grid_cfg_EV.x ) {
        grid_cfg_EV = meso_device->occu_calc.right_peak( 0, gpu_angle_harmonic<1>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_harmonic<1>, cudaFuncCachePreferL1 );
        grid_cfg    = meso_device->occu_calc.right_peak( 0, gpu_angle_harmonic<0>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_angle_harmonic<0>, cudaFuncCachePreferL1 );
    }

    double3 period;
    period.x = ( domain->xperiodic ) ? ( domain->xprd ) : ( 0. );
    period.y = ( domain->yperiodic ) ? ( domain->yprd ) : ( 0. );
    period.z = ( domain->zperiodic ) ? ( domain->zprd ) : ( 0. );

    if( eflag || vflag ) {
        gpu_angle_harmonic<1> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nangletypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_virial(0),
            meso_atom->dev_virial(1),
            meso_atom->dev_virial(2),
            meso_atom->dev_virial(3),
            meso_atom->dev_virial(4),
            meso_atom->dev_virial(5),
            meso_atom->dev_e_angle,
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            dev_k,
            dev_theta0,
            period,
            meso_atom->dev_angle_mapped.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );
    } else {
        gpu_angle_harmonic<0> <<< grid_cfg.x, grid_cfg.y, ( atom->nangletypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force(0),
            meso_atom->dev_force(1),
            meso_atom->dev_force(2),
            meso_atom->dev_virial(0),
            meso_atom->dev_virial(1),
            meso_atom->dev_virial(2),
            meso_atom->dev_virial(3),
            meso_atom->dev_virial(4),
            meso_atom->dev_virial(5),
            meso_atom->dev_e_angle,
            meso_atom->dev_nangle,
            meso_atom->dev_angle_mapped,
            dev_k,
            dev_theta0,
            period,
            meso_atom->dev_angle_mapped.pitch_elem(),
            atom->nangletypes,
            atom->nlocal );
    }
}

