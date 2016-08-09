#ifdef FIX_CLASS

FixStyle(solid_bound/meso,MesoFixSolidBound)

#else

#ifndef LMP_MESO_FIX_SOLID_BOUND
#define LMP_MESO_FIX_SOLID_BOUND

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

enum SolidBoundForceKernel {
	UNSPECIFIED = 0,
	RHO5_RC1_S1
};

class MesoFixSolidBound : public Fix, protected MesoPointers {
public:
	MesoFixSolidBound(class LAMMPS *, int, char **);
	virtual int setmask();
	virtual void init();
	virtual void setup(int);
	virtual void post_force(int);
	virtual void post_integrate();
	virtual void pre_exchange();
	virtual void end_of_step();

protected:
	bool x, y, z;
	SolidBoundForceKernel force_kernel;

	void bounce_back();
	void boundary_force();
};

// mismatched case would give wrong boundary repulsive force!
// for the case of Rho = 5.0, Rc = 1.0 and s = 1.0
struct Rho5rc1s1{
	__host__ __device__ __inline__ double operator () ( double h ) {
		double s = +0.282625;
		s = s * h + -1.39021;
		s = s * h + +2.70259;
		s = s * h + -2.47678;
		s = s * h + +0.863184;
		s = s * h + +0.0664266;
		s = s * h + +0.0247250;
		s = s * h + +0.00856667;
		s = s * h + -0.116714;
		s = s * h * h + 0.0355959;
		return 75.0 * 6.2831853071796 * s;
	}
#if __cplusplus > 199711L
	constexpr static double d = 1.0;
#else
	const static double d = 1.0;
#endif
};

}

#endif

#endif
