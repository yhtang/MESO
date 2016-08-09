#include "error.h"
#include "comm.h"
#include "engine_meso.h"

using namespace LAMMPS_NS;

MesoDevice::MesoDevice( LAMMPS *lmp, int device, std::string profile ):
    Pointers( lmp ),
    MesoPointers( lmp ),
    dummy( false ),
    occu_calc( lmp )
{
    init( device, profile );
    warmup_threshold = 200;
    warmup_progress = 0;
}

MesoDevice::~MesoDevice()
{
#ifdef LMP_MESO_LOG_L3
    cout << "Kernel Summary:" << std::endl;
    KernelIter p;
    for( p = kernel_table.begin() ; p != kernel_table.end() ; p++ ) {
        fprintf( stderr, "Kernel    : %p \n", p->first );
        fprintf( stderr, "numRegs   : %d \n", p->second.numRegs );
        fprintf( stderr, "SharedMem : %.2f KB \n", p->second.sharedSizeBytes / 1024.0 );
        fprintf( stderr, "ConstMem  : %.2f KB \n", p->second.constSizeBytes / 1024.0 );
        fprintf( stderr, "LocalMem  : %.2f KB \n", p->second.localSizeBytes / 1024.0 );
        fprintf( stderr, "BlockSize : %d \n", p->second.maxThreadsPerBlock );
    }
#endif

    print_tuner_stat( std::cerr );

    destroy();
}

int MesoDevice::init( int device, std::string profile )
{
    // detect profiling mode
    if( profile == "all" ) profile_mode = CUDAPROF_ALL;
    else if( profile == "loop" ) profile_mode = CUDAPROF_LOOP;
    else if( profile == "core" ) profile_mode = CUDAPROF_CORE;
    else if( profile.substr( 0, strlen( "interval" ) ) == "interval" ) {
        int n = strlen( "interval" );
        profile_mode = CUDAPROF_INTVL;
        std::string interval = profile.substr( n, profile.size() - n );
        std::replace( interval.begin(), interval.end(), '-', ' ' );
        std::stringstream sstr;
        sstr << interval;
        sstr >> profile_start >> profile_end;
    } else profile_mode = CUDAPROF_NONE;

    // grab device
    device_pool.push_back( device );
    set_device();

    // check capabilities & logging
    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties( &DeviceProp, device );
    device_property.push_back( DeviceProp );
    if( !DeviceProp.canMapHostMemory ) error->one( FLERR, "Mapped memory is not supported by this GPU" );
    if( !DeviceProp.unifiedAddressing ) error->one( FLERR, "Unified Virtual Addressing (UVA) is not supported by this GPU" );
    // create stream
    for( int s = 0 ; s < 16 ; s++ ) {
        stream_pool.insert( stream_pool.end(), CUDAStream( true ) );
    }
    cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
    cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
    cudaDeviceSetLimit( cudaLimitMallocHeapSize, DeviceProp.totalGlobalMem * 0.2 );
    cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 32 * 1024 * 1024 );
    // display information
#if 1
    MPI_Barrier( world );
    int rank;
    MPI_Comm_rank( world, &rank );
    if( rank == 0 ) {
        fprintf( stderr, "<MESO> USER-MESO package for LAMMPS %s %s\n", __DATE__, __TIME__ );
        fprintf( stderr, "<MESO> Copyright (c) 2012-2015 Yu-Hang Tang All rights reserved.\n" );
        fprintf( stderr, "<MESO> Please cite: Tang, Yu-Hang and George Em Karniadakis.\n" );
        fprintf( stderr, "<MESO>              Accelerating dissipative particle dynamics simulations on GPUs:\n" );
        fprintf( stderr, "<MESO>              Algorithms, numerics and applications. Computer Physics Communications\n" );
        fprintf( stderr, "<MESO>              185.11 (2014): 2809-2822.\n" );
    }
    MPI_Barrier( world );
	fprintf( stderr, "<MESO> \tDevice %d, C.C. %d.%d, %.2f GHz, %d kB L2 cache, %d MB GRAM @ %d bit | %d MHz\n", device, DeviceProp.major, DeviceProp.minor,
			 DeviceProp.clockRate / 1048576.0, DeviceProp.l2CacheSize / 1024,
			 DeviceProp.totalGlobalMem / 1024 / 1024, DeviceProp.memoryBusWidth, DeviceProp.memoryClockRate / 1024 );
	MPI_Barrier( world );
#endif

    // start profile if in CUDAPROF_ALL mode
    configure_profiler( -1, -1 );

    return 0;
}

int MesoDevice::destroy()
{
    meso_device->sync_device();

    // stop the profiler
    configure_profiler( -2, -2 );

    for( EventIter i = event_pool.begin() ; i != event_pool.end() ; i++ )
        cudaEventDestroy( i->second.e() );

    for( int i = 0 ; i < stream_pool.size() ; i++ ) cudaStreamDestroy( stream_pool[i] );

    while( mem_table.begin() != mem_table.end() ) free( ( mem_table.begin() )->first );

    stream_pool.clear();
    mem_table.clear();
    kernel_table.clear();
    event_pool.clear();
    texture_ptrs.clear();

    return 0;
}


// show memory usage
void MesoDevice::print_memory_usage( std::ostream &out )
{
    // consolidate usage for each tag
    std::map<std::string, int> usage;
    for( MemIter p = mem_table.begin(); p != mem_table.end(); p++ ) {
        if( usage.find( p->second.tag ) == usage.end() )
            usage[ p->second.tag ]  = p->second.size;
        else
            usage[ p->second.tag ] += p->second.size;
    }
    out << "----------------GPU MEMORY USAGE SUMMARY BEGIN----------------" << std::endl;
    for( std::map<std::string, int>::iterator p = usage.begin(); p != usage.end(); p++ ) {
        out << p->first << '\t' << p->second << std::endl;
    }
    out << "---------------- GPU MEMORY USAGE SUMMARY END ----------------" << std::endl;
}

void MesoDevice::print_tuner_stat( std::ostream &out )
{
    out << "--------------OPENMP THREAD TUNER SUMMARY BEGIN---------------" << std::endl;
    for( TunerIter p = tuner_pool.begin(); p != tuner_pool.end(); p++ ) {
        out << "Tuner " << p->first << " final state: " << p->second.bet() << std::endl;
    }
    out << "-------------- OPENMP THREAD TUNER SUMMARY END ---------------" << std::endl;
}

void MesoDevice::configure_profiler( int ntimestep, int nsteps )
{
    switch( profile_mode ) {
    case CUDAPROF_ALL:
        if( ntimestep == -1 ) profiler_start();
        if( ntimestep == -2 ) profiler_stop();
        break;
    case CUDAPROF_CORE:
        if( ntimestep == nsteps * 0.25 ) profiler_start();
        if( ntimestep == nsteps * 0.75 ) profiler_stop();
        break;
    case CUDAPROF_LOOP:
        if( ntimestep ==      0 ) profiler_start();
        if( ntimestep == nsteps ) profiler_stop();
        break;
    case CUDAPROF_INTVL:
        if( ntimestep == profile_start ) profiler_start();
        if( ntimestep == profile_end ) profiler_stop();
        break;
    default:
        break;
    }
}

void MesoDevice::profiler_start()
{
    meso_device->sync_device();
    cudaProfilerStart();
    fprintf( stderr, "<MESO> Profiler started\n" );
}

void MesoDevice::profiler_stop()
{
    meso_device->sync_device();
    cudaProfilerStop();
    fprintf( stderr, "<MESO> Profiler stopped\n" );
}

void* MesoDevice::texture_ptr( std::string tag, void *ptr )
{
    if( ptr == NULL ) { // query mode
        TextIter p = texture_ptrs.find( tag );
        if( p == texture_ptrs.end() ) return NULL;
        else return p->second;
    } else {
        texture_ptrs[ tag ] = ptr;
        return ptr;
    }
}

CUDAEvent MesoDevice::event( std::string tag )
{
    EventIter p = event_pool.find( tag );
    if( p == event_pool.end() ) {
        #pragma omp critical (MesoDevice_event)
        event_pool.insert( make_pair( tag, CUDAEvent( true ) ) );
        p = event_pool.find( tag );
    }
    return p->second;
}

ThreadTuner& MesoDevice::tuner( std::string tag, std::size_t lower, std::size_t upper )
{
    TunerIter p = tuner_pool.find( tag );
    if( p == tuner_pool.end() ) {
        tuner_pool.insert( make_pair( tag, ThreadTuner( lower, upper, tag ) ) );
        p = tuner_pool.find( tag );
    }
    return p->second;
}

void MesoDevice::free( void *ptr )
{
    MemIter p = mem_table.find( ptr ) ;
    if( p == mem_table.end() ) return;
    if( p->second.memoryType == cudaMemoryTypeDevice ) {
        cudaFree( ptr );
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> GPU[%d] - : %s\n", p->second.device, size2text( p->second.size ).c_str() );
#endif
    } else {
        cudaFreeHost( ptr );
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> HOST - : %s\n", size2text( p->second.size ).c_str() );
#endif
    }
    mem_table.erase( p );
}

std::size_t MesoDevice::query_mem_size( void *ptr )
{
    MemIter p = mem_table.find( ptr ) ;
    if( p == mem_table.end() ) return 0;
    else return p->second.size ;
}

std::size_t MesoDevice::query_mem_pitch( void *ptr )
{
    MemIter p = mem_table.find( ptr ) ;
    if( p == mem_table.end() ) return 0;
    else return p->second.pitch;
}

std::size_t MesoDevice::query_mem_height( void *ptr )
{
    MemIter p = mem_table.find( ptr ) ;
    if( p == mem_table.end() ) return 0;
    else return p->second.height;
}

std::string MesoDevice::size2text( std::size_t size )
{
    std::string Unit;
    float Size;
    if( size < 1024 ) {
        Unit = "B" ;
        Size = size ;
    } else if( size < 1024 * 1024 ) {
        Unit = "KB";
        Size = size / 1024.0 ;
    } else {
        Unit = "MB";
        Size = size / 1024.0 / 1024.0;
    }
    char str[256];
    sprintf( str, "%.1f %s", Size, Unit.c_str() );
    return std::string( str );
}

// return an estimate on the amount of allocated GRAM in bytes
std::size_t MesoDevice::device_allocated() {
	size_t total = 0;
	for( typename std::map<void *, DevicePointerAttr>::iterator i = mem_table.begin(); i != mem_table.end(); i++ ) {
		if ( i->second.memoryType == cudaMemoryTypeDevice ) total += i->second.size;
	}
	return total;
}

// return an estimate on the amount of allocated host memory in bytes
std::size_t MesoDevice::host_allocated() {
	size_t total = 0;
	for( typename std::map<void *, DevicePointerAttr>::iterator i = mem_table.begin(); i != mem_table.end(); i++ ) {
		if ( i->second.memoryType == cudaMemoryTypeHost ) total += i->second.size;
	}
	return total;
}
