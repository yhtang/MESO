#ifdef BOND_CLASS

BondStyle(wlc/edpd/meso,MesoBondWLCeDPD)

#else

#ifndef LMP_MESO_BOND_WLC_EDPD
#define LMP_MESO_BOND_WLC_EDPD

#include "bond.h"
#include "meso.h"

namespace LAMMPS_NS {

class MesoBondWLCeDPD : public Bond, protected MesoPointers
{
public:
	MesoBondWLCeDPD(class LAMMPS *);
	~MesoBondWLCeDPD();
	
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
	double kB;

	void	allocate_gpu();
	void    allocate_cpu();
	int		coeff_alloced;
};

}

#endif

#endif
