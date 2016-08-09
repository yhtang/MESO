#ifdef PAIR_CLASS

PairStyle(sph/meso,MesoPairSPH)

#else

#ifndef LMP_MESO_PAIR_SDPD
#define LMP_MESO_PAIR_SDPD

#include "pair.h"
#include "meso.h"
#include "sph_kernel_meso.h"

namespace LAMMPS_NS {

namespace SPH_ONEBODY {
const static int n_coeff1   = 3;
const static int p_rho0_inv = 0;
const static int p_cs       = 1;
const static int p_B        = 2;
}

namespace SPH_TWOBODY {
const static int n_coeff2 = 4;
const static int p_cut    = 0;
const static int p_cutinv = 1;
const static int p_eta    = 2;
const static int p_n3     = 3;
}

class MesoPairSPH : public Pair, protected MesoPointers {
	friend class FixSPHRhoMeso;
public:
	MesoPairSPH(class LAMMPS *);
	~MesoPairSPH();
	void   compute(int, int);
	void   settings(int, char **);
	void   coeff(int, char **);
	void   init_style();
	double init_one(int, int);
	void   write_restart(FILE *);
	void   read_restart(FILE *);
	void   write_restart_settings(FILE *);
	void   read_restart_settings(FILE *);
	double single(int, int, int, int, double, double, double, double &);

protected:
	int      seed;
	bool     coeff_ready;
	DeviceScalar<r64> dev_coeffs1;
	DeviceScalar<r64> dev_coeffs2;
	std::vector<r64> coeff_table1;
	std::vector<r64> coeff_table2;

	double cut_global;
	double rho0_global;
	double cs_global;
	double eta_global;
	double B_global() const { return rho0_global * cs_global * cs_global / 7.0; }
	double B_one( int i ) const { return rho0[i] * cs[i] * cs[i] / 7.0; }

	double **cut;
	double  *rho0;
	double  *cs;
	double **eta;

	void allocate();
	void prepare_coeff();

	virtual void compute_kernel(int, int, int, int);
};

}

#endif

#endif
