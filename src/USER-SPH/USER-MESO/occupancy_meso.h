#ifndef LMP_MESO_OCCUPANCY
#define LMP_MESO_OCCUPANCY

#include "util_meso.h"

namespace LAMMPS_NS
{

struct OccuRecord {
    OccuRecord()
    {
        threadsPerBlock = 0, activeBlockPerSM = 0, occupancy = 0 ;
    }
    OccuRecord( int t, int b, float o ) : threadsPerBlock( t ), activeBlockPerSM( b ), occupancy( o ) {}
    int threadsPerBlock;
    int activeBlockPerSM;
    float occupancy;
};

struct GPUSpecs {
    int limitThreadsPerWarp;
    int limitWarpsPerMultiprocessor;
    int limitThreadsPerMultiprocessor;
    int limitBlocksPerMultiprocessor;
    int limitTotalSharedMemory;
    int limitTotalRegisters;
    int limitRegsPerThread;
    int maxThreadsPerBlock;
    int warpRegAllocGranularities;
    int myAllocationSize;
    int mySharedMemAllocationSize;
    int myWarpAllocationGranularity;
    int myAllocationGranularity;

    void read( std::string specs )
    {
        char strRegAllocGranu[32] ;
        sscanf( specs.c_str(),
                "%*f %*s %d	%d %d %d %d	%d %d %s %d	%d %d %d %*d %*d %*d %d",
                &limitThreadsPerWarp,
                &limitWarpsPerMultiprocessor,
                &limitThreadsPerMultiprocessor,
                &limitBlocksPerMultiprocessor,
                &limitTotalSharedMemory,
                &limitTotalRegisters,
                &myAllocationSize,
                strRegAllocGranu,
                &limitRegsPerThread,
                &mySharedMemAllocationSize,
                &myWarpAllocationGranularity,
                &maxThreadsPerBlock,
                &warpRegAllocGranularities
              );
        if( !strcmp( strRegAllocGranu, "block" ) ) myAllocationGranularity = 1;
        else myAllocationGranularity = 0;
    }
};

class MesoCUDAOccupancy: protected MesoPointers
{
public:
    typedef void( *void_fptr )();
    typedef std::map<std::pair<void_fptr, std::size_t>, std::vector<OccuRecord> >::iterator OccuIter;

    MesoCUDAOccupancy( LAMMPS * );

    template<typename TYPE>
    int2 left_peak( int device_id, TYPE kernel, size_t dynamic_shmem, enum cudaFuncCache shmemConfig )
    {
        return peak( true, device_id, kernel, dynamic_shmem, shmemConfig );
    }

    template<typename TYPE>
    int2 right_peak( int device_id, TYPE kernel, size_t dynamic_shmem, enum cudaFuncCache shmemConfig )
    {
        return peak( false, device_id, kernel, dynamic_shmem, shmemConfig );
    }

private:
    std::vector<std::string>   gpu_specs_text;
    std::vector<GPUSpecs> gpu_specs;
    std::map<std::pair<void_fptr, std::size_t>, std::vector<OccuRecord> > occupancy_records;

    void resolve_gpu_specs();
    int  get_sm_count( int device_id );
    int2 active_blocks( int device_id, void_fptr kernel, size_t dynamic_shmem, int threadsPerBlock, enum cudaFuncCache shmemConfig );
    const std::vector<OccuRecord>& occupancy( int device_id, void_fptr kernel, size_t dynamic_shmem, enum cudaFuncCache shmemConfig );

    template<typename TYPE>
    int2 peak( bool left, int device_id, TYPE kernel, size_t dynamic_shmem, enum cudaFuncCache shmem_config )
    {
        const std::vector<OccuRecord> &chart = occupancy( device_id, ( void_fptr )kernel, dynamic_shmem, shmem_config );

        float peak_occupancy = 0.;
        for( int i = 0 ; i < chart.size() ; i++ )
            peak_occupancy = max( peak_occupancy, chart[i].occupancy ) ;
        if( left ) {
            for( std::vector<OccuRecord>::const_iterator i = chart.begin() ; i != chart.end() ; i++ ) {
                if( fabs( i->occupancy - peak_occupancy ) < 1e-4 ) {
                    int2 ret;
                    ret.x = i->activeBlockPerSM * get_sm_count( device_id );
                    ret.y = i->threadsPerBlock ;
                    return ret;
                }
            }
        } else {
            for( std::vector<OccuRecord>::const_reverse_iterator i = chart.rbegin() ; i != chart.rend() ; i++ ) {
                if( fabs( i->occupancy - peak_occupancy ) < 1e-4 ) {
                    int2 ret;
                    ret.x = i->activeBlockPerSM * get_sm_count( device_id ) ;
                    ret.y = i->threadsPerBlock ;
                    return ret;
                }
            }
        }
        int2 ret;
        ret.x = chart.back().activeBlockPerSM * get_sm_count( device_id );
        ret.y = chart.back().threadsPerBlock;
        return ret;
    }
};

}

#endif
