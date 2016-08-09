#ifndef LMP_MESO_ATOM_H
#define LMP_MESO_ATOM_H

#include "atom.h"
#include "meso.h"
#include "sort_meso.h"

namespace LAMMPS_NS
{

class MesoAtom : public Atom, protected MesoPointers
{
public:
    MesoAtom( class LAMMPS *lmp );
    ~MesoAtom();

    SortPlan<16, u64, int> sorter;
    class MesoAtomVec *meso_avec;

    int bulk_counting;
    int n_bulk, n_border;
    HostScalar<int> dev_n_bulk;

    int hash_table_size;
    const uint nonce;
    DeviceScalar<int>  dev_map_array;
    DeviceScalar<uint> dev_hash_key;
    DeviceScalar<uint> dev_hash_val;

    PtrDeviceScalar<u64>    dev_perm_key;
    PtrDeviceScalar<int>    dev_permute_from;
    PtrDeviceScalar<int>    dev_permute_to;

    PtrDeviceScalar<int>    dev_tag;
    PtrDeviceScalar<int>    dev_type;
    PtrDeviceScalar<int>    dev_mask;
    PtrDeviceScalar<int>    dev_mole;
    PtrDeviceScalar<r64>    dev_mass;
    PtrDeviceVector<r64, 3> dev_coord;
    PtrDeviceVector<r64, 3> dev_force;
    PtrDeviceVector<r64, 3> dev_veloc;
    PtrDeviceVector<r64, 6> dev_virial;
    PtrDeviceScalar<tagint> dev_image;
    PtrDeviceScalar<r64>    dev_e_bond;
    PtrDeviceScalar<r64>    dev_e_angle;
    PtrDeviceScalar<r64>    dev_e_dihed;
    PtrDeviceScalar<r64>    dev_e_impro;
    PtrDeviceScalar<r64>    dev_e_pair;
    PtrDeviceVector<r32, 3> dev_r_coord;
    PtrDeviceVector<r32, 3> dev_r_veloc;
    PtrDeviceScalar<float4> dev_coord_merged;
    PtrDeviceScalar<float4> dev_veloc_merged;
    PtrDeviceScalar<float4> dev_therm_merged;
    PtrDeviceScalar<r64>    dev_rho;
    PtrDeviceScalar<r64>    dev_Q;
    PtrDeviceScalar<r64>    dev_T;
    PtrHostScalar<int>      hst_borderness;

    PtrDeviceScalar<int>    dev_nbond;
    PtrDevicePitched<int2>  dev_bond;
    PtrDevicePitched<int2>  dev_bond_mapped;
    PtrDevicePitched<r64>   dev_bond_r0;
    PtrDeviceScalar<int>    dev_nangle;
    PtrDevicePitched<int4>  dev_angle;
    PtrDevicePitched<int4>  dev_angle_mapped;
    PtrDevicePitched<r64>   dev_angle_a0;
    PtrDeviceScalar<int>    dev_ndihed;
    PtrDevicePitched<int>   dev_dihed_type;
    PtrDevicePitched<int4>  dev_dihed;
    PtrDevicePitched<int4>  dev_dihed_mapped;
    PtrDevicePitched<int>   devNImprop;
    PtrDevicePitched<int>   devImprops;
    PtrDevicePitched<int>   devImpropType;
    PtrDeviceScalar<int4>   dev_nexcl;
    PtrDeviceScalar<int>    dev_nexcl_full;
    PtrDeviceScalar<int>    dev_excl_table;

    virtual void map_set_device();

    virtual void sort_local();
    virtual void create_avec( const char *, int, char **, char *suffix = NULL );
    virtual void transfer_pre_sort();
    virtual void transfer_pre_post_sort();
    virtual void transfer_post_sort();
    virtual void transfer_pre_border();
    virtual void transfer_post_border();
    virtual void transfer_pre_comm();
    virtual void transfer_post_comm();
    virtual void transfer_pre_exchange();
    virtual void transfer_pre_output();
    void count_bulk_and_border( const std::vector<int> & );

    //  virtual void transfer_post_exchange(); // post_exchange = pre_sort

    // Textures
    TextureObject tex_map_array;
    TextureObject tex_hash_key;
    TextureObject tex_hash_val;
    TextureObject tex_tag;
    TextureObject tex_mass;
    TextureObject tex_rho;
    TextureObject tex_coord_merged;
    TextureObject tex_veloc_merged;

    // dynamic allocation of other textures
    TextureObject& tex_misc( const std::string tag ) {
    	if ( tex_misc_.find(tag) == tex_misc_.end() ) {
    		tex_misc_.insert( std::make_pair( tag, TextureObject( lmp, tag ) ) );
    	}
    	return tex_misc_.find( tag )->second;
    }

protected:
    const double hash_load_factor;

    std::map<std::string,TextureObject> tex_misc_;
    std::vector<int> borderness;
};

}

#endif
