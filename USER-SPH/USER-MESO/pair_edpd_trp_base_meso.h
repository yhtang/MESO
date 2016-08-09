#ifndef LMP_MESO_PAIR_EDPD_TRP_BASE
#define LMP_MESO_PAIR_EDPD_TRP_BASE

#include "meso.h"
#include "pair.h"

namespace LAMMPS_NS {

namespace PNIPAM_COEFFICIENTS {
const static int n_coeff  = 10;
const static int p_cut    = 0;
const static int p_cutinv = 1;
const static int p_a0     = 2;
const static int p_gamma  = 3;
const static int p_cv     = 4;
const static int p_sq2kpa = 5;
const static int p_theta  = 6;
const static int p_da     = 7;
const static int p_omega  = 8;
const static int p_s      = 9;
}

class MesoPairEDPDTRPBase: public Pair {
public:
	friend class MesoFixBoundaryFcTRP;
	friend class MesoFixBoundaryFdTRP;
	friend class MesoFixBoundaryFcTRPSpecial;
	friend class MesoFixBoundaryFdTRPSpecial;
	friend class MesoComputeEcEDPD;
	friend class FixEDPDEnergyConvert;

protected:
	MesoPairEDPDTRPBase(class LAMMPS *);
	virtual ~MesoPairEDPDTRPBase() {}
	virtual void prepare_coeff() = 0;

	bool              coeff_ready;
	DeviceScalar<r64> dev_coefficients;
};

}

#endif
