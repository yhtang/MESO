#ifdef INTEGRATE_CLASS

IntegrateStyle(mvv/meso,ModifiedVerlet)
IntegrateStyle(verlet/meso,ModifiedVerlet)

#else

#ifndef LMP_MESO_MVV
#define LMP_MESO_MVV

#include "integrate.h"
#include "meso.h"

namespace LAMMPS_NS {

class ModifiedVerlet : public Integrate, protected MesoPointers
{
public:
	ModifiedVerlet(class LAMMPS *, int, char **);
	virtual ~ModifiedVerlet() {}
	virtual void init();
	virtual void setup();
	virtual void setup_minimal(int);
	virtual void run(int);
	virtual void cleanup() {}

	virtual void check_error(int linenum, const char filename[]);

protected:
	virtual void force_clear( AtomAttribute::Descriptor range = AtomAttribute::LOCAL );
};

}

#endif

#endif
