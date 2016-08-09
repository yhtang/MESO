#ifdef FIX_CLASS

FixStyle(rho/meso,FixSPHRhoMeso)

#else

#ifndef LMP_MESO_FIX_SPH_RHO
#define LMP_MESO_FIX_SPH_RHO

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixSPHRhoMeso : public Fix, protected MesoPointers
{
public:
	FixSPHRhoMeso(LAMMPS *lmp, int narg, char **arg);

	virtual int setmask();
	virtual void pre_force(int);
	virtual void setup_pre_force(int);
	virtual int pack_comm(int, int *, double *, int, int *);
	virtual void unpack_comm(int, int, double *);

protected:
	class MesoPairSPH *pair;

	void prepare_coeff();
};

}

#endif

#endif
