#ifndef LMP_MESO_BIN
#define LMP_MESO_BIN

#include "meso.h"


namespace LAMMPS_NS
{

class MesoBin: protected Pointers, protected MesoPointers
{
public:
    MesoBin( class LAMMPS * );
    ~MesoBin();
    void alloc_bins();

    DeviceScalar<uint> dev_bin_id;
    DeviceScalar<int>  dev_atm_id;
    DeviceScalar<int>  dev_bin_location;
    DeviceScalar<int>  dev_bin_size;
    DeviceScalar<int>  dev_bin_size_local;
    DeviceScalar<int>  dev_bin_isghost; // 1:ghost, 0:local. p.s. by aligning bin boundary with my box make sure bins are 'pure'

    TextureObject tex_atm_id;
};

}

#endif
