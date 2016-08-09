#ifdef PAIR_CLASS

PairStyle(dpd/tableforce/meso,MesoPairDPDTableForce)

#else

#ifndef LMP_MESO_PAIR_DPD_TABLEFORCE
#define LMP_MESO_PAIR_DPD_TABLEFORCE

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS {

// tabulated force stored as CUDA 1D layered textures

namespace TABLEFORCE_COEFFICIENTS {
const static int n_coeff  = 33;
const static int p_cut    = 0;
const static int p_cutsq  = 1;
const static int p_cutinv = 2;
const static int p_gamma  = 3;
const static int p_sigma  = 4;
}

class MesoPairDPDTableForce : public Pair, protected MesoPointers {
 public:
	MesoPairDPDTableForce(class LAMMPS *);
	~MesoPairDPDTableForce();
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
	std::vector<r32> coeff_table;

	cudaArray_t         dev_table;
	cudaTextureObject_t tex_table;
	std::vector<r32>    hst_table;
	int                 table_length;
	float2              table_scaling;

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
