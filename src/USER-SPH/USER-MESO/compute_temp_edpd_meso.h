#ifdef COMPUTE_CLASS

ComputeStyle(temp/edpd/meso,MesoComputeTempEDPD)

#else

#ifndef LMP_MESO_COMPUTE_TEMP_EDPD
#define LMP_MESO_COMPUTE_TEMP_EDPD

#include "compute.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoComputeTempEDPD : public Compute, protected MesoPointers {
 public:
	MesoComputeTempEDPD(class LAMMPS *, int, char **);
	~MesoComputeTempEDPD();
	virtual void init() {}
	virtual void setup();
	virtual double compute_scalar();

 private:
	DeviceScalar<r64>    t;
	DeviceScalar<int>    c;
};

}

#endif

#endif
