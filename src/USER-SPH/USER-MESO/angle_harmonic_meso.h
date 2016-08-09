#ifdef ANGLE_CLASS

AngleStyle(harmonic/meso,MesoAngleHarmonic)

#else

#ifndef LMP_MESO_ANGLE_HARMONIC
#define LMP_MESO_ANGLE_HARMONIC

#include "angle_harmonic.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoAngleHarmonic : public AngleHarmonic, protected MesoPointers
{
public:
	MesoAngleHarmonic(class LAMMPS *);
	~MesoAngleHarmonic();
	
	void compute( int eflag, int vflag );
protected:
	DeviceScalar<r64> dev_k;
	DeviceScalar<r64> dev_theta0;

	void	alloc_coeff();
	int		coeff_alloced;
};

}

#endif

#endif
