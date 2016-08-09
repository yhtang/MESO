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
#include "bond_wlc_meso.h"
#include "neighbor_meso.h"

using namespace LAMMPS_NS;

MesoBondWLC::MesoBondWLC( LAMMPS *lmp ):
    Bond(lmp),
    MesoPointers( lmp ),
    dev_lda( lmp, "MesoBondWLC::dev_lda" ),
    dev_Lsp( lmp, "MesoBondWLC::dev_Lsp" )
{
    kBT = 1.0;
    coeff_alloced = 0;
}

MesoBondWLC::~MesoBondWLC()
{
	if (setflag) memory->destroy(setflag);
}

void MesoBondWLC::allocate_gpu()
{
    if( coeff_alloced ) return;

    coeff_alloced = 1;
    int n = atom->nbondtypes;
    dev_lda.grow( n + 1, false, false );
    dev_Lsp.grow( n + 1, false, false );
    dev_lda.upload( lda.data(), n + 1, meso_device->stream() );
    dev_Lsp.upload( Lsp.data(), n + 1, meso_device->stream() );
}

void MesoBondWLC::allocate_cpu()
{
	allocated = 1;
	int n = atom->nbondtypes;
	lda.resize(n+1);
	Lsp.resize(n+1);
	memory->create(setflag,n+1,"bond:setflag");
	for (int i = 1; i <= n; i++) setflag[i] = 0;
}

template<int evflag>
__global__ void gpu_bond_wlc(
    texobj tex_coord,
    r64*  __restrict force_x,
    r64*  __restrict force_y,
    r64*  __restrict force_z,
    r64*  __restrict virial_xx,
    r64*  __restrict virial_yy,
    r64*  __restrict virial_zz,
    r64*  __restrict virial_xy,
    r64*  __restrict virial_xz,
    r64*  __restrict virial_yz,
    r64*  __restrict e_bond,
    int*  __restrict nbond,
    int2* __restrict bonds,
    r64*  __restrict lda_global,
    r64*  __restrict Lsp_global,
    const r64 kBT,
    const double3 period,
    const int padding,
    const int n_type,
    const int n_local )
{
    extern __shared__ r64 shared_data[];
    r64 *lda = &shared_data[0];
    r64 *Lsp = &shared_data[n_type + 1];
    for( int i = threadIdx.x ; i < n_type + 1 ; i += blockDim.x ) {
        lda[i] = 1. / lda_global[i];
        Lsp[i] = 1. / Lsp_global[i];
    }
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_local ; i += gridDim.x * blockDim.x ) {
        int n = nbond[i];
        f3i coord1 = tex1Dfetch<float4>( tex_coord, i );

        r64 fx = 0.0, fy = 0.0, fz = 0.0;
        r64 e = 0.0 ;

        for( int p = 0 ; p < n ; p++ ) {
            int j    = bonds[ i + p * padding ].x;
            int type = bonds[ i + p * padding ].y;

            f3i coord2 = tex1Dfetch<float4>( tex_coord, j );
            r64 dx = minimum_image( coord2.x - coord1.x, period.x ) ;
            r64 dy = minimum_image( coord2.y - coord1.y, period.y ) ;
            r64 dz = minimum_image( coord2.z - coord1.z, period.z ) ;

            r64 rsq   = dx * dx + dy * dy + dz * dz ;
            r64 rinv  = rsqrt( rsq ) ;
            r64 r     = rinv * rsq ;
            r64 R     = r / Lsp[type];
            r64 fbond = kBT / lda[type] * ( 0.25 / power<2>( 1 - R ) - 0.25 + R );

            fx += dx * fbond ;
            fy += dy * fbond ;
            fz += dz * fbond ;
            if( evflag ) e += kBT / ( 4. * lda[type] ) * ( Lsp[type]*Lsp[type]/( Lsp[type] - r ) - r + (2.*r*r) / Lsp[type] );
        }

        force_x[i]   += fx;
        force_y[i]   += fy;
        force_z[i]   += fz;
        if( evflag ) {
            virial_xx[i] += coord1.x * fx ;
            virial_yy[i] += coord1.y * fy ;
            virial_zz[i] += coord1.z * fz ;
            virial_xy[i] += coord1.x * fy ;
            virial_xz[i] += coord1.x * fz ;
            virial_yz[i] += coord1.y * fz ;
            e_bond[i]     = e * 0.5;
        }
    }
}

void MesoBondWLC::compute( int eflag, int vflag )
{
    if( !coeff_alloced ) allocate_gpu();

    static GridConfig grid_cfg, grid_cfg_EV;
    if( !grid_cfg_EV.x ) {
        grid_cfg_EV = meso_device->occu_calc.right_peak( 0, gpu_bond_wlc<1>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_wlc<1>, cudaFuncCachePreferL1 );
        grid_cfg    = meso_device->occu_calc.right_peak( 0, gpu_bond_wlc<0>, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), cudaFuncCachePreferL1 );
        cudaFuncSetCacheConfig( gpu_bond_wlc<0>, cudaFuncCachePreferL1 );
    }

    double3 period;
    period.x = ( domain->xperiodic ) ? ( domain->xprd ) : ( 0. );
    period.y = ( domain->yperiodic ) ? ( domain->yprd ) : ( 0. );
    period.z = ( domain->zperiodic ) ? ( domain->zprd ) : ( 0. );

    if( eflag || vflag ) {
        gpu_bond_wlc<1> <<< grid_cfg_EV.x, grid_cfg_EV.y, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force[0],
            meso_atom->dev_force[1],
            meso_atom->dev_force[2],
            meso_atom->dev_virial[0],
            meso_atom->dev_virial[1],
            meso_atom->dev_virial[2],
            meso_atom->dev_virial[3],
            meso_atom->dev_virial[4],
            meso_atom->dev_virial[5],
            meso_atom->dev_e_bond,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            dev_lda,
            dev_Lsp,
            kBT,
            period,
            meso_atom->dev_bond_mapped.pitch(),
            atom->nbondtypes,
            atom->nlocal );
    } else {
        gpu_bond_wlc<0> <<< grid_cfg.x, grid_cfg.y, ( atom->nbondtypes + 1 ) * 2 * sizeof( r64 ), meso_device->stream() >>> (
            meso_atom->tex_coord_merged,
            meso_atom->dev_force[0],
            meso_atom->dev_force[1],
            meso_atom->dev_force[2],
            meso_atom->dev_virial[0],
            meso_atom->dev_virial[1],
            meso_atom->dev_virial[2],
            meso_atom->dev_virial[3],
            meso_atom->dev_virial[4],
            meso_atom->dev_virial[5],
            meso_atom->dev_e_bond,
            meso_atom->dev_nbond,
            meso_atom->dev_bond_mapped,
            dev_lda,
            dev_Lsp,
            kBT,
            period,
            meso_atom->dev_bond_mapped.pitch(),
            atom->nbondtypes,
            atom->nlocal );
    }
}

void MesoBondWLC::settings(int narg, char **arg)
{
	for(int i = 0 ; i < narg ; i++) {
		if ( !strcmp( arg[i], "kBT" ) ) {
			kBT = force->numeric(FLERR,arg[++i]);
		} else {
			error->warning(FLERR,"unknown parameter for bond style wlc/edpd");
		}
	}
}

/* ----------------------------------------------------------------------
   parse coefficients from input script
------------------------------------------------------------------------- */

void MesoBondWLC::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate_cpu();

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

  double lda_one = force->numeric(FLERR,arg[1]);
  double Lsp_one = force->numeric(FLERR,arg[2]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    lda[i] = lda_one;
    Lsp[i] = Lsp_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double MesoBondWLC::equilibrium_distance(int i)
{
  return 0.; // WLC is constantly attractive
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void MesoBondWLC::write_restart(FILE *fp)
{
  fwrite(&lda[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&Lsp[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void MesoBondWLC::read_restart(FILE *fp)
{
  allocate_cpu();

  if (comm->me == 0) {
    fread(&lda[1],sizeof(double),atom->nbondtypes,fp);
    fread(&Lsp[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&lda[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&Lsp[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void MesoBondWLC::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g\n",i,lda[i],Lsp[i]);
}

/* ---------------------------------------------------------------------- */

double MesoBondWLC::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  error->warning(FLERR,"<MESO> MesoBondWLC::single not implemented");
	return 0.;
}
