#ifdef PAIR_CLASS

PairStyle(dpd/polyforce/meso,MesoPairDPDPolyForce)

#else

#ifndef LMP_MESO_PAIR_DPD_POLYFORCE
#define LMP_MESO_PAIR_DPD_POLYFORCE

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

// polynomial in the form of
// p(s) = a0 + a1 * s + a2 * s^2 + ...
// where s = ( 1 - r / rc )
// instead of s = r OR r / rc because this makes it easier for s to approach 0 when r goes to rc

namespace PEPTOID_COEFFICIENTS {
const static int polynomial_maxlen = 32;
const static int n_coeff  = 5;
const static int p_cut    = 0;
const static int p_cutsq  = 1;
const static int p_cutinv = 2;
const static int p_gamma  = 3;
const static int p_sigma  = 4;
}

class MesoPairDPDPolyForce : public Pair, protected MesoPointers {
 public:
	MesoPairDPDPolyForce(class LAMMPS *);
	~MesoPairDPDPolyForce();
	void   compute(int, int);
	void   compute_bulk(int, int);
	void   compute_border(int, int);
	void   settings(int, char **);
	void   coeff(int, char **);
	void   init_style();
	double init_one(int, int);
	void   write_restart(FILE *);
	void   read_restart(FILE *);
	void   write_restart_settings(FILE *);
	void   read_restart_settings(FILE *);

 protected:
	int      seed;
	bool     coeff_ready;
	DeviceScalar<r32> dev_coefficients;
	DeviceScalar<r32> dev_polynomial;
	std::vector<r32> coeff_table;
	std::vector<r32> polynomial; // polynomial format: 1 (order) + polynomial_maxlen * coefficients (from higher order to lowest)

	float   cut_global;
	float **cut;
	float **cut_inv;
	float **gamma;
	float **sigma;

	class RanMars *random;

	virtual void allocate();
	virtual void prepare_coeff();
	virtual void compute_kernel(int, int, int, int);
	virtual uint seed_now();
};

}

#endif

#endif
