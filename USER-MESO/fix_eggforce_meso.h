#ifdef FIX_CLASS

FixStyle(eggforce/meso,FixEggForce)

#else

#ifndef LMP_MESO_FIX_EGG_FORCE
#define LMP_MESO_FIX_EGG_FORCE

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixEggForce : public Fix, protected MesoPointers
{
public:
	FixEggForce(LAMMPS *lmp, int narg, char **arg);

	virtual int setmask();
	virtual void post_force(int);

protected:
	int ref_group, ref_groupbit;
	double rc, sigma, bodyforce, rho0;
};

}

#endif

#endif
