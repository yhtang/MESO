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
//
//#include "mpi.h"
//#include "string.h"
//#include "stdlib.h"
//#include "update.h"
//#include "domain.h"
//#include "modify.h"
//#include "fix.h"
//#include "force.h"
//#include "pair.h"
//#include "bond.h"
//#include "angle.h"
//#include "dihedral.h"
//#include "improper.h"
//#include "kspace.h"
//#include "error.h"
//
//#include "engine_meso.h"
//#include "atom_meso.h"
//#include "compute_pressure_meso.h"
//
//using namespace LAMMPS_NS;
////
//MesoComputePressure::MesoComputePressure(LAMMPS *lmp, int narg, char **arg) :
//  Compute(lmp, narg, arg)
//{
//  if (narg < 4) error->all(__FILE__,__LINE__,"Illegal compute pressure command");
//  if (igroup) error->all(__FILE__,__LINE__,"Compute pressure must use group all");
//
//  scalar_flag = vector_flag = 1;
//  size_vector = 6;
//  extscalar = 0;
//  extvector = 0;
//  pressflag = 1;
//  timeflag = 1;
//
//  // store temperature ID used by pressure computation
//  // insure it is valid for temperature computation
//
//  int n = strlen(arg[3]) + 1;
//  id_pre = new char[n];
//  strcpy(id_pre,arg[3]);
//
//  int icompute = modify->find_compute(id_pre);
//  if (icompute < 0) error->all(__FILE__,__LINE__,"Could not find compute pressure temp ID");
//  if (modify->compute[icompute]->tempflag == 0)
//      error->all(__FILE__,__LINE__,"Compute pressure temp ID does not compute temperature");
//
//  // process optional args
//
//  if (narg == 4) {
//      keflag = 1;
//      pairflag = 1;
//      bondflag = angleflag = dihedralflag = improperflag = 1;
//      fixflag = 1;
//  } else {
//      keflag = 0;
//      pairflag = 0;
//      bondflag = angleflag = dihedralflag = improperflag = 0;
//      fixflag = 0;
//      int iarg = 4;
//      while (iarg < narg) {
//          if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
//          else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
//          else if (strcmp(arg[iarg],"bond") == 0) bondflag = 1;
//          else if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
//          else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
//          else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
//          else if (strcmp(arg[iarg],"fix") == 0) fixflag = 1;
//          else if (strcmp(arg[iarg],"virial") == 0)
//          {
//              pairflag = 1;
//              bondflag = angleflag = dihedralflag = improperflag = 1;
//              fixflag = 1;
//          }
//          else error->all(__FILE__,__LINE__,"Illegal compute pressure command");
//          iarg++;
//      }
//  }
//
//  vector = new double[6];
//
//  VBuf = cuda_engine->MallocHost<r64>( "MesoComputePressure::VBuf", 6 );
//}
//
//MesoComputePressure::~MesoComputePressure()
//{
//  delete [] vector;
//}
//
//void MesoComputePressure::init()
//{
//  boltz = force->boltz;
//  nktv2p = force->nktv2p;
//  dimension = domain->dimension;
//
//  // set temperature compute, must be done in init()
//  // fixes could have changed or compute_modify could have changed it
//
//  int icompute = modify->find_compute(id_pre);
//  if (icompute < 0) error->all(__FILE__,__LINE__,"Could not find compute pressure temp ID");
//  temperature = modify->compute[icompute];
//
////    // detect contributions to virial
////    // vptr points to all virial[6] contributions
////    // OBSOLETE because in this CUDA implementation all virial contributions are written
////    // to the same vector located in MesoAtomVec
////
////    delete [] vptr;
////    nvirial = 0;
////    vptr = NULL;
////
////    if (pairflag && force->pair) nvirial++;
////    if (bondflag && atom->molecular && force->bond) nvirial++;
////    if (angleflag && atom->molecular && force->angle) nvirial++;
////    if (dihedralflag && atom->molecular && force->dihedral) nvirial++;
////    if (improperflag && atom->molecular && force->improper) nvirial++;
////    if (fixflag)
////        for (int i = 0; i < modify->nfix; i++)
////            if (modify->fix[i]->virial_flag) nvirial++;
////
////    if (nvirial) {
////        vptr = new double*[nvirial];
////        nvirial = 0;
////        if (pairflag && force->pair) vptr[nvirial++] = force->pair->virial;
////        if (bondflag && force->bond) vptr[nvirial++] = force->bond->virial;
////        if (angleflag && force->angle) vptr[nvirial++] = force->angle->virial;
////        if (dihedralflag && force->dihedral) vptr[nvirial++] = force->dihedral->virial;
////        if (improperflag && force->improper) vptr[nvirial++] = force->improper->virial;
////        if (fixflag)
////            for (int i = 0; i < modify->nfix; i++)
////    if (modify->fix[i]->virial_flag)
////        vptr[nvirial++] = modify->fix[i]->virial;
////    }
//}
//
///* ----------------------------------------------------------------------
//   compute total pressure, averaged over Pxx, Pyy, Pzz
//------------------------------------------------------------------------- */
//
//double MesoComputePressure::compute_scalar()
//{
//  invoked_scalar = update->ntimestep;
//  if (update->vflag_global != invoked_scalar)
//      error->all(__FILE__,__LINE__,"Virial was not tallied on needed timestep");
//
//  // invoke temperature it it hasn't been already
//
//  double t;
//  if (keflag)
//  {
//      if (temperature->invoked_scalar == update->ntimestep) t = temperature->scalar;
//      else t = temperature->compute_scalar();
//  }
//
//  if (dimension == 3)
//  {
//      inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
//      virial_compute(3,3);
//      if (keflag)
//          scalar = (temperature->dof * boltz * t + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
//      else
//          scalar = (virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
//  }
//  else
//  {
//      inv_volume = 1.0 / (domain->xprd * domain->yprd);
//      virial_compute(2,2);
//      if (keflag)
//          scalar = (temperature->dof * boltz * t + virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
//      else
//          scalar = (virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
//  }
//
//  return scalar;
//}
//
///* ----------------------------------------------------------------------
//   compute pressure tensor
//   assume KE tensor has already been computed
//------------------------------------------------------------------------- */
//
//void MesoComputePressure::compute_vector()
//{
//  invoked_vector = update->ntimestep;
//  if (update->vflag_global != invoked_vector)
//      error->all(__FILE__,__LINE__,"Virial was not tallied on needed timestep");
//
//  // invoke temperature it it hasn't been already
//
//  double *ke_tensor;
//  if (keflag) {
//      if (temperature->invoked_vector != update->ntimestep)
//          temperature->compute_vector();
//      ke_tensor = temperature->vector;
//  }
//
//  if (dimension == 3)
//  {
//      inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
//      virial_compute(6,3);
//      if (keflag)
//      {
//          for (int i = 0; i < 6; i++)
//              vector[i] = (ke_tensor[i] + virial[i]) * inv_volume * nktv2p;
//      }
//      else
//          for (int i = 0; i < 6; i++)
//              vector[i] = virial[i] * inv_volume * nktv2p;
//  }
//  else
//  {
//      inv_volume = 1.0 / (domain->xprd * domain->yprd);
//      virial_compute(4,2);
//      if (keflag)
//      {
//          vector[0] = (ke_tensor[0] + virial[0]) * inv_volume * nktv2p;
//          vector[1] = (ke_tensor[1] + virial[1]) * inv_volume * nktv2p;
//          vector[3] = (ke_tensor[3] + virial[3]) * inv_volume * nktv2p;
//      }
//      else
//      {
//          vector[0] = virial[0] * inv_volume * nktv2p;
//          vector[1] = virial[1] * inv_volume * nktv2p;
//          vector[3] = virial[3] * inv_volume * nktv2p;
//      }
//  }
//}
//
//void MesoComputePressure::virial_compute(int n, int ndiag)
//{
//  // sum contributions to virial from forces and fixes
//  r64 *devVirial[] = {    cuda_atom->cuda_avec->devVirial[0], cuda_atom->cuda_avec->devVirial[1], cuda_atom->cuda_avec->devVirial[2],
//                          cuda_atom->cuda_avec->devVirial[3], cuda_atom->cuda_avec->devVirial[4], cuda_atom->cuda_avec->devVirial[5] };
//  size_t threads_per_block = 1024 ;
//  for(int i = 0 ; i < n ; i++ )
//  {
//      gpu_reduce_sum_host<<< 1, threads_per_block, 0, cuda_engine->stream() >>> (
//          devVirial[i],
//          VBuf + i,
//          atom->nlocal
//      );
//  }
//  cuda_engine->stream().sync();
//
//  // sum virial across procs
//
//  MPI_Allreduce(VBuf,virial,n,MPI_DOUBLE,MPI_SUM,world);
//
//  // RAINN: KSpace virial not supported in CUDA DPD
//
//  // LJ long-range tail correction
//
//  if (force->pair && force->pair->tail_flag)
//      for (int i = 0; i < ndiag; i++) virial[i] += force->pair->ptail * inv_volume;
//}
