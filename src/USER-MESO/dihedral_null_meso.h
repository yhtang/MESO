#ifdef DIHEDRAL_CLASS

DihedralStyle(null/meso,MesoDihedralNull)

#else

#ifndef LMP_MESO_DIHEDRAL_NULL
#define LMP_MESO_DIHEDRAL_NULL

#include "dihedral_harmonic.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoDihedralNull : public DihedralHarmonic, protected MesoPointers
{
public:
	MesoDihedralNull(class LAMMPS *);
	~MesoDihedralNull();
	
	void compute( int eflag, int vflag );
protected:
};

}

#endif

#endif
