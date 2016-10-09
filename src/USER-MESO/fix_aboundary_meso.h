#ifdef FIX_CLASS

FixStyle(aboundary/meso,FixArbitraryBoundary)

#else

#ifndef LMP_MESO_FIX_EDPD_ARBITRARY_BOUNDARY
#define LMP_MESO_FIX_EDPD_ARBITRARY_BOUNDARY

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixArbitraryBoundary : public Fix, protected MesoPointers
{
public:
	FixArbitraryBoundary(LAMMPS *lmp, int narg, char **arg);

	virtual int setmask();
	virtual void post_force(int);
	virtual void end_of_step();

protected:
	int wall_group, wall_groupbit;
	double rho0, rc, sigma;
};

}

#endif

#endif
