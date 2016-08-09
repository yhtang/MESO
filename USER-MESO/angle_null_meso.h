#ifdef ANGLE_CLASS

AngleStyle(null/meso,MesoAngleNull)

#else

#ifndef LMP_MESO_ANGLE_NULL
#define LMP_MESO_ANGLE_NULL

#include "angle_harmonic.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoAngleNull : public AngleHarmonic, protected MesoPointers
{
public:
	MesoAngleNull(class LAMMPS *);
	void compute( int eflag, int vflag );
};

}

#endif

#endif
