#ifdef BOND_CLASS

BondStyle(harmonic/meso,MesoBondHarmonic)

#else

#ifndef LMP_MESO_BOND_HARMONIC
#define LMP_MESO_BOND_HARMONIC

#include "bond_harmonic.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoBondHarmonic : public BondHarmonic, protected MesoPointers
{
public:
	MesoBondHarmonic(class LAMMPS *);
	~MesoBondHarmonic();
	
	void	compute( int eflag, int vflag );
protected:
	DeviceScalar<r64> dev_k;
	DeviceScalar<r64> dev_r0;

	void	alloc_coeff();
	int		coeff_alloced;
};

}

#endif

#endif
