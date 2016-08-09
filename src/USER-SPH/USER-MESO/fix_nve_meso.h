#ifdef FIX_CLASS

FixStyle(nve/meso,FixNVEMeso)

#else

#ifndef LMP_MESO_FIX_NVE
#define LMP_MESO_FIX_NVE

#include "fix_nve.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixNVEMeso : public Fix, protected MesoPointers
{
public:
	FixNVEMeso(class LAMMPS *, int, char **);
	virtual void init();
	virtual int setmask();
	virtual void initial_integrate(int);
	virtual void final_integrate();
	virtual void reset_dt();
protected:
	double dtv, dtf, dtT, Cv;
};

}

#endif

#endif
