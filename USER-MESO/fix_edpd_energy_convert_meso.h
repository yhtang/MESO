#ifdef FIX_CLASS

FixStyle(ecvt/edpd/meso,FixEDPDEnergyConvert)

#else

#ifndef LMP_MESO_FIX_EDPD_ENERGY_CONVERT
#define LMP_MESO_FIX_EDPD_ENERGY_CONVERT

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class FixEDPDEnergyConvert : public Fix, protected MesoPointers
{
public:
	FixEDPDEnergyConvert(LAMMPS *lmp, int narg, char **arg);

	virtual int setmask();
	virtual void initial_integrate(int);
	virtual int pack_comm(int, int *, double *, int, int *);
	virtual void unpack_comm(int, int, double *);

protected:
	int jgroup, jgroupbit;
	class MesoPairEDPDTRPBase *pair;
	DeviceScalar<r32> dev_dE;
	DeviceScalar<r64> dev_Q_in;
};

}

#endif

#endif
