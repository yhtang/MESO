#ifdef COMPUTE_CLASS

ComputeStyle(pe/meso,MesoComputePE)

#else

#ifndef LMP_MESO_COMPUTE_PE
#define LMP_MESO_COMPUTE_PE

#include "meso.h"
#include "memory_meso.h"
#include "compute.h"

namespace LAMMPS_NS {

class MesoComputePE : public Compute, protected MesoPointers
{
public:
	MesoComputePE(LAMMPS *, int , char **);
	~MesoComputePE() {}
	void init() {}
	double compute_scalar();

protected:
	HostScalar<r64> per_atom_energy;

private:
	int pairflag,bondflag,angleflag,dihedralflag,improperflag;
};

}

#endif

#endif
