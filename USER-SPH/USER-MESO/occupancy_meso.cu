#include "engine_meso.h"
#include "occupancy_meso.h"

using namespace LAMMPS_NS;

MesoCUDAOccupancy::MesoCUDAOccupancy( LAMMPS *lmp ) : MesoPointers( lmp )
{
    gpu_specs_text.push_back( "1.0    sm_10    32    24     768    8    16384    8192    256    block    124    512    2     512    16384        0        0      0" );
    gpu_specs_text.push_back( "1.1    sm_11    32    24     768    8    16384    8192    256    block    124    512    2     512    16384        0        0      0" );
    gpu_specs_text.push_back( "1.2    sm_12    32    32    1024    8    16384    16384   512    block    124    512    2     512    16384        0        0      0" );
    gpu_specs_text.push_back( "1.3    sm_13    32    32    1024    8    16384    16384   512    block    124    512    2     512    16384        0        0      0" );
    gpu_specs_text.push_back( "2.0    sm_20    32    48    1536    8    49152    32768   64      warp     63    128    2    1024    49152    16384        0     64" );
    gpu_specs_text.push_back( "2.1    sm_21    32    48    1536    8    49152    32768   64      warp     63    128    2    1024    49152    16384        0     64" );
    gpu_specs_text.push_back( "3.0    sm_30    32    64    2048   16    49152    65536   256     warp     63    256    4    1024    49152    16384    32768    256" );
    gpu_specs_text.push_back( "3.5    sm_35    32    64    2048   16    49152    65536   256     warp    255    256    4    1024    49152    16384    32768    256" );
    gpu_specs_text.push_back( "3.7    sm_37    32    64    2048   16   114688   131072   256     warp    255    256    4    1024   114688    81920    98304    256" );
    gpu_specs_text.push_back( "5.0    sm_50    32    64    2048   32    65536    65536   256     warp    255    256    4    1024    49152    16384    32768    256" );
    gpu_specs_text.push_back( "5.2    sm_52    32    64    2048   32    98304    65536   256     warp    255    256    4    1024    98304    16384    32768    256" );
    gpu_specs_text.push_back( "5.3    sm_53    32    64    2048   32    65536    32768   256     warp    255    256    4    1024    98304    16384    32768    256" );
    // experimental
    gpu_specs_text.push_back( "6.0    sm_60    32    64    2048   32    65536    65536   256     warp    255    256    4    1024    98304    16384    32768    256" );
    gpu_specs_text.push_back( "6.1    sm_61    32    64    2048   32    65536    65536   256     warp    255    256    4    1024    98304    16384    32768    256" );
}

int MesoCUDAOccupancy::get_sm_count( int device_id )
{
    return meso_device->device_property[ device_id ].multiProcessorCount;
}

void MesoCUDAOccupancy::resolve_gpu_specs()
{
    if( gpu_specs.size() == meso_device->device_pool.size() ) return;

    gpu_specs.resize( meso_device->device_pool.size() );
    for( int did = 0; did < gpu_specs.size() ; did++ ) {
        bool find = false;
        char sm_version[32];
        sprintf( sm_version, "%d.%d", meso_device->device_property[did].major, meso_device->device_property[did].minor );
        std::string smVersion = sm_version ;
        for( int i = 0 ; i < gpu_specs_text.size() ; i++ ) {
            // matching specs for current architecture
            if( gpu_specs_text[i].substr( 0, 3 ) == smVersion ) {
                gpu_specs[did].read( gpu_specs_text[i] );
                find = true;
                break;
            }
        }
        if( !find ) {
            fprintf( stderr, "[CDEV] cannot find matching specs for the current GPU #%d %s.\n", did, sm_version );
        }
    }
}

const std::vector<OccuRecord>& MesoCUDAOccupancy::occupancy( int device_id, void_fptr kernel, size_t dynamic_shmem, enum cudaFuncCache shmemConfig )
{
    OccuIter p_rec = occupancy_records.find( std::make_pair( kernel, dynamic_shmem ) );
    if( p_rec == occupancy_records.end() ) {
        std::vector<OccuRecord> chart;

        resolve_gpu_specs();
        GPUSpecs &specs = gpu_specs[device_id];

        int kernelRegCount;
        int kernelSharedMemory;
        int configTotalSharedMemory;
        switch( shmemConfig ) {
        case cudaFuncCachePreferL1    :
            configTotalSharedMemory = 16 * 1024 ;
            break;
        case cudaFuncCachePreferShared:
            configTotalSharedMemory = 48 * 1024 ;
            break;
        case cudaFuncCachePreferEqual :
            configTotalSharedMemory = 32 * 1024 ;
            break;
        default:
            configTotalSharedMemory = 48 * 1024 ;
            break;
        }

        // get kernel handle for querying attributes
        std::map<void( * )(), cudaFuncAttributes >::iterator p = meso_device->kernel_table.find( kernel ) ;
        if( p == meso_device->kernel_table.end() ) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes( &attr, kernel );
            meso_device->kernel_table[ kernel ] = attr ;
            p = meso_device->kernel_table.find( kernel );
        }
        kernelRegCount = p->second.numRegs ;
        kernelSharedMemory = p->second.sharedSizeBytes + dynamic_shmem ;

        // calculate occupancy array with varying block size, fixed: reg usage & sharemem size
        chart.reserve( specs.maxThreadsPerBlock / specs.limitThreadsPerWarp );
        for( int MyThreadCount = specs.limitThreadsPerWarp ; MyThreadCount <= specs.maxThreadsPerBlock ; MyThreadCount += specs.limitThreadsPerWarp ) {
            // CUDA Occupancy Calculator, B38-B40
            int MyWarpsPerBlock, MyRegsPerBlock, MySharedMemPerBlock ;
            MyWarpsPerBlock = ceiling( MyThreadCount, specs.limitThreadsPerWarp ) / specs.limitThreadsPerWarp ;
            if( specs.myAllocationGranularity == 1 )
                MyRegsPerBlock = ceiling( ceiling( MyWarpsPerBlock, specs.myWarpAllocationGranularity ) * kernelRegCount * specs.limitThreadsPerWarp, specs.myAllocationSize );
            else
                MyRegsPerBlock = ceiling( kernelRegCount * specs.limitThreadsPerWarp, specs.myAllocationSize ) * ceiling( MyWarpsPerBlock, specs.myWarpAllocationGranularity );
            MySharedMemPerBlock = kernelSharedMemory ;

            // CUDA Occupancy Calculator, D38-D40
            int limitBlocksDueToWarps, limitBlocksDueToRegs, limitBlocksDueToSMem;
            limitBlocksDueToWarps = min( specs.limitBlocksPerMultiprocessor, specs.limitWarpsPerMultiprocessor / MyWarpsPerBlock ) ;
            if( kernelRegCount > specs.limitRegsPerThread ) limitBlocksDueToRegs = 0;
            else {
                if( kernelRegCount > 0 ) limitBlocksDueToRegs = specs.limitTotalRegisters / MyRegsPerBlock ;
                else limitBlocksDueToRegs = specs.limitBlocksPerMultiprocessor ;
            }
            if( MySharedMemPerBlock > 0 ) limitBlocksDueToSMem = configTotalSharedMemory / MySharedMemPerBlock ;
            else limitBlocksDueToSMem = specs.limitBlocksPerMultiprocessor ;

            // Calculate occupancy
            int ActiveBlocks  = min( min( limitBlocksDueToWarps, limitBlocksDueToRegs ), limitBlocksDueToSMem ) ;
            int ActiveWarps   = ActiveBlocks * MyWarpsPerBlock ;
            __attribute__( ( unused ) ) int ActiveThreads = ActiveWarps  * specs.limitThreadsPerWarp ;
            float Occupancy  = ( double )ActiveWarps / specs.limitWarpsPerMultiprocessor ;
            chart.push_back( OccuRecord( MyThreadCount, ActiveBlocks, Occupancy ) );
        }

        occupancy_records[ std::make_pair( kernel, dynamic_shmem ) ] = chart;
        p_rec = occupancy_records.find( std::make_pair( kernel, dynamic_shmem ) );
    }
    return p_rec->second;
}

// .x: active block per SM
// .y: active block on entire GPU
int2 MesoCUDAOccupancy::active_blocks( int device_id, void_fptr kernel, size_t dynamic_shmem, int threads_per_block, enum cudaFuncCache shmemConfig )
{
    resolve_gpu_specs();
    GPUSpecs &specs = gpu_specs[device_id];

    int kernelRegCount;
    int kernelSharedMemory;
    int configTotalSharedMemory;
    switch( shmemConfig ) {
    case cudaFuncCachePreferL1    :
        configTotalSharedMemory = 16 * 1024 ;
        break;
    case cudaFuncCachePreferShared:
        configTotalSharedMemory = 48 * 1024 ;
        break;
    case cudaFuncCachePreferEqual :
        configTotalSharedMemory = 32 * 1024 ;
        break;
    default:
        configTotalSharedMemory = 48 * 1024 ;
        break;
    }

    // get kernel handle for querying attributes
    MesoDevice::KernelIter p = meso_device->kernel_table.find( kernel ) ;
    if( p == meso_device->kernel_table.end() ) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes( &attr, kernel );
        meso_device->kernel_table[ kernel ] = attr ;
        p = meso_device->kernel_table.find( kernel );
    }
    kernelRegCount = p->second.numRegs ;
    kernelSharedMemory = p->second.sharedSizeBytes + dynamic_shmem ;

    // CUDA Occupancy Calculator, B38-B40
    int MyWarpsPerBlock, MyRegsPerBlock, MySharedMemPerBlock ;
    MyWarpsPerBlock = ceiling( threads_per_block, specs.limitThreadsPerWarp ) / specs.limitThreadsPerWarp ;
    if( specs.myAllocationGranularity == 1 )
        MyRegsPerBlock = ceiling( ceiling( MyWarpsPerBlock, specs.myWarpAllocationGranularity ) * kernelRegCount * specs.limitThreadsPerWarp, specs.myAllocationSize );
    else
        MyRegsPerBlock = ceiling( kernelRegCount * specs.limitThreadsPerWarp, specs.myAllocationSize ) * ceiling( MyWarpsPerBlock, specs.myWarpAllocationGranularity );
    MySharedMemPerBlock = kernelSharedMemory ;

    // CUDA Occupancy Calculator, D38-D40
    int limitBlocksDueToWarps, limitBlocksDueToRegs, limitBlocksDueToSMem;
    limitBlocksDueToWarps = min( specs.limitBlocksPerMultiprocessor, specs.limitWarpsPerMultiprocessor / MyWarpsPerBlock ) ;
    if( kernelRegCount > specs.limitRegsPerThread ) limitBlocksDueToRegs = 0;
    else {
        if( kernelRegCount > 0 ) limitBlocksDueToRegs = specs.limitTotalRegisters / MyRegsPerBlock ;
        else limitBlocksDueToRegs = specs.limitBlocksPerMultiprocessor ;
    }
    if( MySharedMemPerBlock > 0 ) limitBlocksDueToSMem = configTotalSharedMemory / MySharedMemPerBlock ;
    else limitBlocksDueToSMem = specs.limitBlocksPerMultiprocessor ;

    // Calculate occupancy
    int Active = min( min( limitBlocksDueToWarps, limitBlocksDueToRegs ), limitBlocksDueToSMem ) ;
    return make_int2( Active, Active * meso_device->device_property[0].multiProcessorCount );
}

