#ifdef COMPUTE_CLASS

ComputeStyle(temp/meso,MesoComputeTemp)

#else

#ifndef LMP_MESO_COMPUTE_TEMP
#define LMP_MESO_COMPUTE_TEMP

#include "compute.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoComputeTemp : public Compute, protected MesoPointers {
 public:
	MesoComputeTemp(class LAMMPS *, int, char **);
	~MesoComputeTemp();
	virtual void init() {}
	virtual void setup();
	virtual double compute_scalar();

 private:
	int fix_dof;
	double tfactor;
	HostScalar<r64>    t;
	DeviceScalar<r64> per_atom_eK;

	virtual void dof_compute();
};

}

#endif

#endif
