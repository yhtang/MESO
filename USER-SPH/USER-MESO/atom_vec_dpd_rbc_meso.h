#ifdef ATOM_CLASS

AtomStyle(dpd/rbc/meso,AtomVecDPDRBC)

#else

#ifndef LMP_MESO_ATOM_VEC_DPD_RBC
#define LMP_MESO_ATOM_VEC_DPD_RBC

#include "atom_vec_rbc.h"
#include "atom_vec_dpd_molecular_meso.h"

namespace LAMMPS_NS {

class AtomVecDPDRBC: public AtomVecRBC, public AtomVecDPDMolecular
{
public:
	AtomVecDPDRBC(LAMMPS *);
	~AtomVecDPDRBC() {}

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
	DevicePitched<r64>  dev_bond_r0;

	DeviceScalar<r64>   dev_e_angle;
	DeviceScalar<int>   dev_nangle;
	DevicePitched<int4> dev_angle;
	DevicePitched<int4> dev_angle_mapped;
	DevicePitched<r64>  dev_angle_a0;

	DeviceScalar<r64>   dev_e_dihed;
	DeviceScalar<int>   dev_ndihed;
	DevicePitched<int>  dev_dihed_type;
	DevicePitched<int4> dev_dihed;
	DevicePitched<int4> dev_dihed_mapped;

	Pinned<int> dev_nbond_pinned;
	Pinned<int> dev_bond_atom_pinned;
	Pinned<int> dev_bond_type_pinned;
	Pinned<double> dev_bond_r0_pinned;

	Pinned<int> dev_nangle_pinned;
	Pinned<int> dev_angle_atom1_pinned;
	Pinned<int> dev_angle_atom2_pinned;
	Pinned<int> dev_angle_atom3_pinned;
	Pinned<int> dev_angle_type_pinned;
	Pinned<double> dev_angle_a0_pinned;

	Pinned<int> dev_ndihed_pinned;
	Pinned<int> dev_dihed_atom1_pinned;
	Pinned<int> dev_dihed_atom2_pinned;
	Pinned<int> dev_dihed_atom3_pinned;
	Pinned<int> dev_dihed_atom4_pinned;
	Pinned<int> dev_dihed_type_pinned;

	virtual void transfer_impl( std::vector<CUDAEvent> &events, AtomAttribute::Descriptor per_atom_prop, TransferDirection direction, int p_beg, int n_atom, int p_stream, int p_inc, int* permute_to, int* permute_from, int action, bool streamed);
	CUDAEvent transfer_bond (TransferDirection direction, int* permute_from, int p_beg, int n_transfer, CUDAStream stream, int action);
	CUDAEvent transfer_angle(TransferDirection direction, int *permute_to, int p_beg, int n_transfer, CUDAStream stream, int action);
	CUDAEvent transfer_dihed(TransferDirection direction, int *permute_to, int p_beg, int n_transfer, CUDAStream stream, int action);
};

}

#endif

#endif
