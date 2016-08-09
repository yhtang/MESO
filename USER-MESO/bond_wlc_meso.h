#ifdef BOND_CLASS

BondStyle(wlc/meso,MesoBondWLC)

#else

#ifndef LMP_MESO_BOND_WLC
#define LMP_MESO_BOND_WLC

#include "bond.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoBondWLC : public Bond, protected MesoPointers
{
public:
	MesoBondWLC(class LAMMPS *);
	~MesoBondWLC();
	
	void settings(int, char **);
	void coeff(int, char **);
	void compute(int, int);
	double equilibrium_distance(int);
	void write_restart(FILE *);
	void read_restart(FILE *);
	void write_data(FILE *);
	double single(int, double, int, int, double &);

protected:
	DeviceScalar<r64> dev_lda;
	DeviceScalar<r64> dev_Lsp;
	std::vector<double> lda, Lsp;
	double kBT;

	void	allocate_gpu();
	void    allocate_cpu();
	int		coeff_alloced;
};

}

#endif

#endif
