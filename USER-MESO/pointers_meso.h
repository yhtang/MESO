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

// Pointers class contains ptrs to master copy of
//   fundamental LAMMPS class ptrs stored in lammps.h
// every LAMMPS class inherits from Pointers to access lammps.h ptrs
// these variables are auto-initialized by Pointer class constructor
// *& variables are really pointers to the pointers in lammps.h
// & enables them to be accessed directly in any class, e.g. atom->x
// specialized version for LAMMPS

#ifndef LMP_MESO_POINTERS
#define LMP_MESO_POINTERS

#include "lammps.h"

namespace LAMMPS_NS
{

class MesoPointers
{
public:
    MesoPointers( LAMMPS *ptr ) :
        meso_atom( *( ( MesoAtom** )( &ptr->atom ) ) ),
        meso_comm( *( ( MesoComm** )( &ptr->comm ) ) ),
        meso_domain( *( ( MesoDomain** )( &ptr->domain ) ) ),
        meso_error( *( ( MesoError** )( &ptr->error ) ) ),
        meso_neighbor( *( ( MesoNeighbor** )( &ptr->neighbor ) ) ),
        meso_timer( *( ( MesoTimer** )( &ptr->timer ) ) ),
        meso_device( ptr->meso_device )
    {
    }
    MesoPointers( MesoPointers *ptr ) :
        meso_atom( ptr->meso_atom ),
        meso_comm( ptr->meso_comm ),
        meso_domain( ptr->meso_domain ),
        meso_error( ptr->meso_error ),
        meso_neighbor( ptr->meso_neighbor ),
        meso_timer( ptr->meso_timer ),
        meso_device( ptr->meso_device ) {}
    virtual ~MesoPointers() {}

protected:
    class MesoAtom     *&meso_atom;
    class MesoComm     *&meso_comm;
    class MesoDomain   *&meso_domain;
    class MesoError    *&meso_error;
    class MesoNeighbor *&meso_neighbor;
    class MesoTimer    *&meso_timer;
    class MesoDevice   *&meso_device;
};

}

#endif
