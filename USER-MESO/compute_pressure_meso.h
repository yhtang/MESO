///* ----------------------------------------------------------------------
//   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
//   http://lammps.sandia.gov, Sandia National Laboratories
//   Steve Plimpton, sjplimp@sandia.gov
//
//   Copyright (2003) Sandia Corporation.  Under the terms of Contract
//   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
//   certain rights in this software.  This software is distributed under
//   the GNU General Public License.
//
//   See the README file in the top-level LAMMPS directory.
//------------------------------------------------------------------------- */
//
//#ifdef COMPUTE_CLASS
//
//ComputeStyle(pressure/meso,ComputePressure)
//
//#else
//
//#ifndef LMP_MESO_COMPUTE_PRESSURE
//#define LMP_MESO_COMPUTE_PRESSURE
//
//#include "compute.h"
//
//namespace LAMMPS_NS {
//
//class MesoComputePressure : public Compute {
// public:
//  MesoComputePressure(class LAMMPS *, int, char **);
//  ~MesoComputePressure();
//  void init();
//  double compute_scalar();
//  void compute_vector();
//
// private:
//  double boltz,nktv2p,inv_volume;
//  int dimension;
//  Compute *temperature;
//  double virial[6];
//  int keflag,pairflag,bondflag,angleflag,dihedralflag,improperflag;
//  int fixflag;
//
//  void virial_compute(int, int);
//
//  r64 *VBuf;
//};
//
//}
//
//#endif
//
//#endif
