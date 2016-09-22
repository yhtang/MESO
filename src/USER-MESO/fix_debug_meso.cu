///* ----------------------------------------------------------------------
//   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
//   http://lammps.sandia.gov, Sandia National Laboratories
//   Steve Plimpton, sjplimp@sandia.gov
//
//   Copyright (2003) Sandia Corporation.   Under the terms of Contract
//   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
//   certain rights in this software.   This software is distributed under
//   the GNU General Public License.
//
//   See the README file in the top-level LAMMPS directory.
//------------------------------------------------------------------------- */
//#include "mpi.h"
//#include "stdio.h"
//#include "string.h"
//#include "force.h"
//#include "update.h"
//#include "respa.h"
//#include "error.h"
//
//#include "atom_vec_meso.h"
//#include "fix_debug_meso.h"
//#include "engine_meso.h"
//#include "atom_meso.h"
//
//using namespace LAMMPS_NS;
////
//MesoFixDebug::MesoFixDebug(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
//{
//  nevery = 1;
//  monitor_target = 0;
//  for( int i = 0 ; i < narg ; i++ )
//  {
//      if (!strcmp(arg[i],"velocity") ) monitor_target = 0;
//      else if (!strcmp(arg[i],"force") ) monitor_target = 1;
//  }
//}
//
//void MesoFixDebug::init()
//{
//  p = 0;
//  Total = cuda_engine->MallocDevice<r32>( "MesoFixDebug::Total", N_SLOT * 3 );
//  cudaMemsetAsync( Total, 0, N_SLOT*3*sizeof(r32), cuda_engine->stream() );
//  if ( monitor_target == 0 ) fout.open("fix_debug.log.velocity");
//  else if ( monitor_target == 1 ) fout.open("fix_debug.log.force");
//}
//
//MesoFixDebug::~MesoFixDebug()
//{
//  dump();
//  if ( Total ) cuda_engine->Free(Total);
//}
//int MesoFixDebug::setmask()
//{
//  int mask = 0;
////  mask |= INITIAL_INTEGRATE;
////  mask |= POST_INTEGRATE;
////  mask |= PRE_EXCHANGE;
////  mask |= PRE_NEIGHBOR;
////  mask |= PRE_FORCE;
////  mask |= POST_FORCE;
////  mask |= FINAL_INTEGRATE;
//  mask |= END_OF_STEP;
//  return mask;
//}
//
//void MesoFixDebug::initial_integrate(int vflag) { vtotal("II"); }
//void MesoFixDebug::post_integrate()             { vtotal("PI"); }
//void MesoFixDebug::pre_exchange()               { vtotal("PE"); }
//void MesoFixDebug::pre_neighbor()               { vtotal("PN"); }
//void MesoFixDebug::pre_force(int vflag)         { vtotal("PRF"); }
//void MesoFixDebug::post_force(int vflag)        { vtotal("POF"); }
//void MesoFixDebug::final_integrate()            { vtotal("FI"); }
//void MesoFixDebug::end_of_step()                { vtotal(""); }
//
//__global__ void gpuFixDebug(
//  r32* __restrict Total,
//  r64* __restrict X,
//  r64* __restrict Y,
//  r64* __restrict Z,
//  const int  n_atom )
//{
//  for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x )
//  {
//      if ( i < n_atom&0XFFFFFFE0 )
//      {
//          r32 x = __warp_sum( (r32) X[i] );
//          r32 y = __warp_sum( (r32) Y[i] );
//          r32 z = __warp_sum( (r32) Z[i] );
//          if ( __laneid() == 0 )
//          {
//              atomic_add( Total+0, x );
//              atomic_add( Total+1, y );
//              atomic_add( Total+2, z );
//          }
//      }
//      else
//      {
//          atomic_add( Total+0, X[i] );
//          atomic_add( Total+1, Y[i] );
//          atomic_add( Total+2, Z[i] );
//      }
//  }
//}
//
//void MesoFixDebug::vtotal( string tag )
//{
//  static GridConfig grid_cfg;
//  if ( !grid_cfg.x )
//  {
//      grid_cfg = cuda_engine->OccuCalc.right_peak( 0, (void*)gpuFixDebug, 0, cudaFuncCachePreferL1 );
//      cudaFuncSetCacheConfig( gpuFixDebug, cudaFuncCachePreferL1 );
//  }
//
//  Tags.push_back( tag );
//  nT.push_back(update->ntimestep);
//
//  r64 *ptrX, *ptrY, *ptrZ;
//  switch (monitor_target)
//  {
//      case 0: ptrX =cuda_atom->cuda_avec->devVelo[0]; ptrY =cuda_atom->cuda_avec->devVelo[1]; ptrZ =cuda_atom->cuda_avec->devVelo[2]; break;
//      case 1: ptrX =cuda_atom->cuda_avec->devForce[0]; ptrY =cuda_atom->cuda_avec->devForce[1]; ptrZ =cuda_atom->cuda_avec->devForce[2]; break;
//  }
//
//  gpuFixDebug<<< grid_cfg.x, grid_cfg.y, 0, cuda_engine->stream() >>>(
//      Total + (p++)*3,
//      ptrX, ptrY, ptrZ,
//      atom->nlocal );
//
//  if ( p == N_SLOT ) dump();
//}
//
//void MesoFixDebug::dump()
//{
//  vector<r32> total( N_SLOT * 3, 0.0 );
//  cudaMemcpy( &total[0], Total, N_SLOT*3*sizeof(32), cudaMemcpyDefault);
//  for(int i = 0 ; i < Tags.size(); i++ )
//  {
//      fout<<nT[i]<<'\t'
//          <<Tags[i]<<'\t'
//          <<total[i*3+0]/atom->nlocal<<'\t'
//          <<total[i*3+1]/atom->nlocal<<'\t'
//          <<total[i*3+2]/atom->nlocal<<endl;
//  }
//  fout.flush();
//  cudaMemsetAsync( Total, 0, N_SLOT*3*sizeof(r32), cuda_engine->stream() );
//  p=0;
//  Tags.clear();
//  nT.clear();
//}
