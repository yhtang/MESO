#ifdef COMPUTE_CLASS

ComputeStyle(ec/edpd/meso,MesoComputeEcEDPD)

#else

#ifndef LMP_MESO_COMPUTE_EC_EDPD
#define LMP_MESO_COMPUTE_EC_EDPD

#include "compute.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoComputeEcEDPD : public Compute, protected MesoPointers {
public:
	MesoComputeEcEDPD(class LAMMPS *, int, char **);
	~MesoComputeEcEDPD();
	virtual void init() {}
	virtual void setup();
	virtual double compute_scalar();

protected:
	DeviceScalar<r64> dev_energy;
	DeviceScalar<int> dev_ninter;
	HostScalar<r64> hst_energy;
	HostScalar<int> hst_ninter;
	int jgroup, jgroupbit;
	bool per_pair;
	class MesoPairEDPDTRPBase *pair;
};

}

#endif

#endif
