/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef _ACCELERATOR_MESO_H
#define _ACCELERATOR_MESO_H

// true interface to USER-MESO
// used when USER-MESO is installed

#ifdef LMP_USER_MESO

#include "atom_meso.h"
#include "comm_meso.h"
#include "domain_meso.h"
#include "error_meso.h"
#include "engine_meso.h"
#include "neighbor_meso.h"
#include "timer_meso.h"

#else

// dummy interface to USER-MESO
// needed for compiling when USER-MESO is not installed

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "neighbor.h"
#include "timer.h"

namespace LAMMPS_NS
{

class CUDAEngine
{
public:
	const bool dummy;
	CUDAEngine( LAMMPS *, int, string ) : dummy(true) {}
    ~CUDAEngine() {}
};

class MesoAtom : public Atom {
 public:
	MesoAtom(class LAMMPS *lmp) : Modify(lmp) {}
	~MesoAtom() {}
};

class MesoComm : public Comm
{
public:
	MesoComm(class LAMMPS *lmp) : Comm(lmp) {}
	~MesoComm() {}
};

class MesoDomain : public Domain
{
public:
	MesoDomain(class LAMMPS *lmp) : Domain(lmp) {}
	~MesoDomain() {}
};

class MesoError : public Error
{
public:
	MesoError(class LAMMPS *lmp) : Error(lmp) {}
	~MesoError() {}
};

class MesoNeighbor : public Neighbor {
public:
	MesoNeighbor(class LAMMPS *lmp) : Neighbor(lmp) {}
	~MesoNeighbor() {}
};

class MesoTimer : public Timer {
public:
	MesoTimer(class LAMMPS *lmp) : Timer(lmp) {}
	~MesoTimer() {}
};

}

#endif
#endif
