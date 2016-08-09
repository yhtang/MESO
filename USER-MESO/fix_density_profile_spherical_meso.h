#ifdef FIX_CLASS

FixStyle(density/spherical/meso,MesoFixDensitySpherical)

#else

#ifndef LMP_MESO_FIX_DENSITY_SPHERICAL
#define LMP_MESO_FIX_DENSITY_SPHERICAL

#include "fix.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoFixDensitySpherical : public Fix, protected MesoPointers
{
public:
	MesoFixDensitySpherical(LAMMPS *lmp, int narg, char **arg);
	~MesoFixDensitySpherical();

	virtual int setmask();
	virtual void setup(int);
	virtual void post_integrate();

protected:
	DeviceScalar<float>   dev_com;
	DevicePitched<int>    dev_polar_grids1, dev_polar_grids2;
	DeviceScalar<uint>    dev_polar_grid_size1, dev_polar_grid_size2;
	DeviceScalar<r64>     dev_polar_grid_meanr1, dev_polar_grid_meanr2;
	DeviceScalar<r64>     dev_density_profile;
	int n_grid_t, n_grid_p, n_grid;
	int target_groupbit;
	r64 da; // size of polar grids in fraction of PI
	int n_half_bin;
	r64 maxr, dr;
	int every, window;
	std::string filename;

protected:
	int n_sample;
	bigint last_dump_time;

	void dump(bigint);
};

}

#endif

#endif
