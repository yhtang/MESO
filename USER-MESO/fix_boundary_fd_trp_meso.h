#ifdef FIX_CLASS

FixStyle(boundary/fd/trp/meso,MesoFixBoundaryFdTRP)

#else

#ifndef LMP_MESO_FIX_BOUNDARY_FD_TRP
#define LMP_MESO_FIX_BOUNDARY_FD_TRP

#include "fix.h"
#include "meso.h"

// TRP = thermo-responsive polymer

namespace LAMMPS_NS {

class MesoFixBoundaryFdTRP : public Fix, protected MesoPointers {
public:
	MesoFixBoundaryFdTRP(class LAMMPS *, int, char **);
	~MesoFixBoundaryFdTRP();
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);

private:
	int wall_type;
	double T0;
	double nx, ny, nz, H, cut;
	double v0x, v0y, v0z;
	// poly + A0:
	// coefficient for the functional form: A0/h + polynomial(h)
	double A0;
	HostScalar<double> poly;

	class MesoPairEDPDTRPBase *pair;

	void prepare_coeff();


};

}

#endif

#endif
