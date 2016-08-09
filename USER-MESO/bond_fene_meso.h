#ifdef BOND_CLASS

BondStyle(fene/meso,MesoBondFENE)

#else

#ifndef LMP_MESO_BOND_FENE
#define LMP_MESO_BOND_FENE

#include "bond_fene.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoBondFENE : public BondFENE, protected MesoPointers
{
public:
	MesoBondFENE(class LAMMPS *);
	~MesoBondFENE();

	void	compute( int eflag, int vflag );
protected:
	DeviceScalar<r64> dev_k;
	DeviceScalar<r64> dev_r0;
	DeviceScalar<r64> dev_epsilon;
	DeviceScalar<r64> dev_sigma;

	void	alloc_coeff();
	int		coeff_alloced;
};

}

#endif

#endif
