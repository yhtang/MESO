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
    DeviceScalar<uint> dev_map_array;
    DeviceScalar<uint> dev_hash_key;
    DeviceScalar<uint> dev_hash_val;

    MesoMemory<DeviceScalar,u64>    dev_perm_key;
    MesoMemory<DeviceScalar,int>    dev_permute_from;
    MesoMemory<DeviceScalar,int>    dev_permute_to;

    MesoMemory<DeviceScalar,int>    dev_tag;
    MesoMemory<DeviceScalar,int>    dev_type;
    MesoMemory<DeviceScalar,int>    dev_mask;
    MesoMemory<DeviceScalar,int>    dev_mole;
    MesoMemory<DeviceScalar,r64>    dev_mass;
    MesoMemory<DeviceVector,r64>    dev_coord;
    MesoMemory<DeviceVector,r64>    dev_force;
    MesoMemory<DeviceVector,r64>    dev_veloc;
    MesoMemory<DeviceVector,r64>    dev_virial;
    MesoMemory<DeviceScalar,tagint> dev_image;
    MesoMemory<DeviceScalar,r64>    dev_e_bond;
    MesoMemory<DeviceScalar,r64>    dev_e_angle;
    MesoMemory<DeviceScalar,r64>    dev_e_dihed;
    MesoMemory<DeviceScalar,r64>    dev_e_impro;
    MesoMemory<DeviceScalar,r64>    dev_e_pair;
    MesoMemory<DeviceVector,r32>    dev_r_coord;
    MesoMemory<DeviceVector,r32>    dev_r_veloc;
    MesoMemory<DeviceScalar,float4> dev_coord_merged;
    MesoMemory<DeviceScalar,float4> dev_veloc_merged;
    MesoMemory<DeviceScalar,float4> dev_therm_merged;
    MesoMemory<DeviceScalar,r64>    dev_rho;
    MesoMemory<DeviceScalar,r64>    dev_Q;
    MesoMemory<DeviceScalar,r64>    dev_T;
    MesoMemory<HostScalar,int>      hst_borderness;

    MesoMemory<DeviceScalar,int>    dev_nbond;
    MesoMemory<DevicePitched,int2>  dev_bond;
    MesoMemory<DevicePitched,int2>  dev_bond_mapped;
    MesoMemory<DevicePitched,r64>   dev_bond_r0;
    MesoMemory<DeviceScalar,int>    dev_nangle;
    MesoMemory<DevicePitched,int4>  dev_angle;
    MesoMemory<DevicePitched,int4>  dev_angle_mapped;
    MesoMemory<DevicePitched,r64>   dev_angle_a0;
    MesoMemory<DeviceScalar,int>    dev_ndihed;
    MesoMemory<DevicePitched,int>   dev_dihed_type;
    MesoMemory<DevicePitched,int4>  dev_dihed;
    MesoMemory<DevicePitched,int4>  dev_dihed_mapped;
    MesoMemory<DevicePitched,int>   devNImprop;
    MesoMemory<DevicePitched,int>   devImprops;
    MesoMemory<DevicePitched,int>   devImpropType;
    MesoMemory<DeviceScalar,int4>   dev_nexcl;
    MesoMemory<DeviceScalar,int>    dev_nexcl_full;
    MesoMemory<DeviceScalar,int>    dev_excl_table;

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
