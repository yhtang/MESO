#ifdef ATOM_CLASS

AtomStyle(dpd/bond/meso,AtomVecDPDBond)

#else

#ifndef LMP_MESO_ATOM_VEC_DPD_BOND
#define LMP_MESO_ATOM_VEC_DPD_BOND

#include "atom_vec_bond.h"
#include "atom_vec_dpd_molecular_meso.h"

namespace LAMMPS_NS {

class AtomVecDPDBond: public AtomVecBond, public AtomVecDPDMolecular
{
public:
	AtomVecDPDBond(LAMMPS *);
	~AtomVecDPDBond() {}

	virtual void data_atom_target(int, double*, int, char**);
	virtual void grow(int);
	virtual void grow_reset();
	virtual void grow_cpu(int);
	virtual void grow_device(int);
	virtual void pin_host_array();
	virtual void unpin_host_array();

protected:
	DeviceScalar<r64>   dev_e_bond;
	DeviceScalar<int>   dev_nbond;
	DevicePitched<int2> dev_bond;
	DevicePitched<int2> dev_bond_mapped;

	Pinned<int> dev_nbond_pinned;
	Pinned<int> dev_bond_atom_pinned;
	Pinned<int> dev_bond_type_pinned;

	virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed);
	CUDAEvent transfer_bond(TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action);
};

}

#endif

#endif
