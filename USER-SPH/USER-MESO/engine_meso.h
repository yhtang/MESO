#ifndef LMP_MESO_ENGINE
#define LMP_MESO_ENGINE

#include "pointers.h"
#include "autotuner_meso.h"
#include "math_meso.h"
#include "type_meso.h"
#include "util_meso.h"
#include "pointers_meso.h"
#include "occupancy_meso.h"
#include "engine_util_meso.h"

namespace LAMMPS_NS
{

struct DevicePointerAttr : public cudaPointerAttributes {
    std::size_t size;
    std::size_t pitch;
    std::size_t height;
    std::string tag;
    DevicePointerAttr()
    {
        size  = 0;
        pitch = 0;
        height = 0;
    }
};

class MesoDevice: protected Pointers, protected MesoPointers
{
    friend class MesoCUDAOccupancy;

protected:
    std::vector<int>                         device_pool;
    std::vector<CUDAStream>              stream_pool;
    std::vector<cudaDeviceProp>           device_property;
    std::map<void *, DevicePointerAttr>   mem_table;
    std::map<void_fptr, cudaFuncAttributes> kernel_table;
    std::map<std::string, CUDAEvent>           event_pool;
    std::map<std::string, void *>              texture_ptrs;
    std::map<std::string, ThreadTuner>         tuner_pool;
    int                              warmup_threshold, warmup_progress;

public:
    MesoDevice( LAMMPS *, int, std::string );
    ~MesoDevice();

    int                              profile_mode, profile_start, profile_end;
    const bool                       dummy;
    MesoCUDAOccupancy                occu_calc;

    // types
    typedef std::map<void *, DevicePointerAttr>::iterator   MemIter;
    typedef std::map<void_fptr, cudaFuncAttributes>::iterator KernelIter;
    typedef std::map<std::string, CUDAEvent>::iterator           EventIter;
    typedef std::map<std::string, void *>::iterator              TextIter;
    typedef std::map<std::string, ThreadTuner>::iterator           TunerIter;

    // interaces
    int    init( int Device, std::string Profile );
    int    destroy();
    void   free( void *ptr );
    void   configure_profiler( int, int );
    void   profiler_start();
    void   profiler_stop();
    void   print_memory_usage( std::ostream &out );
    void   print_tuner_stat( std::ostream &out );
    std::size_t query_mem_size( void *ptr );
    std::size_t query_mem_pitch( void *ptr );
    std::size_t query_mem_height( void *ptr );
    CUDAEvent event( std::string tag );
    ThreadTuner& tuner( std::string tag, std::size_t lower = 1, std::size_t upper = omp_get_max_threads() );
    void*     texture_ptr( std::string tag, void *ptr = NULL );

    // light utilities
    inline CUDAStream stream( int i = 0 )
    {
        return stream_pool[ i % stream_pool.size() ];
    }
    inline void sync_device()
    {
        watch( ( cudaDeviceSynchronize() ) );
    }
    inline void set_device()
    {
        cudaSetDevice( device_pool[0] );
    }
    inline bool warmed_up() const
    {
        return warmup_progress > warmup_threshold;
    }
    inline void next_step()
    {
        warmup_progress++;
    }

    template<typename FTYPE> int2 configure_kernel( FTYPE kernel, int dsmem = 0, bool prefer_large_block = true, cudaFuncCache cache_pref = cudaFuncCachePreferNone )
    {
        cudaFuncSetCacheConfig( kernel, cache_pref );
        if( prefer_large_block )
            return meso_device->occu_calc.right_peak( 0, kernel, dsmem, cache_pref );
        else
            return meso_device->occu_calc.left_peak( 0, kernel, dsmem, cache_pref );
    }

    template<typename FTYPE> std::size_t query_block_size( FTYPE *fptr )
    {
        void_fptr ptr = ( void_fptr ) fptr;
        KernelIter p = kernel_table.find( ptr ) ;
        if( p == kernel_table.end() ) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes( &attr, ptr );
            kernel_table[ ptr ] = attr ;
            if( attr.maxThreadsPerBlock == 0 ) fprintf( stderr, "<MESO> Querying block size returned 0, something might have crashed somewhere else.\n" );
            return attr.maxThreadsPerBlock;
        } else return p->second.maxThreadsPerBlock ;
    }

    std::size_t device_allocated();
    std::size_t host_allocated();
    std::string size2text( std::size_t size );

    template <typename TYPE> TYPE *malloc_device( std::string tag, std::size_t nElem )
    {
        TYPE *ptr;
        if( cudaMalloc( &ptr, nElem * sizeof( TYPE ) ) != cudaSuccess ) {
            fprintf( stderr, "<MESO> %s size = %s for %s, allocated = %s\n",
                     cudaGetErrorString( cudaPeekAtLastError() ),
                     size2text( nElem * sizeof( TYPE ) ).c_str(),
                     tag.c_str(),
                     size2text( device_allocated() ).c_str() );
            for( typename std::map<void *, DevicePointerAttr>::iterator i = mem_table.begin(); i != mem_table.end(); i++ ) {
				if ( i->second.memoryType == cudaMemoryTypeDevice ) {
		            fprintf( stderr, "<MESO> allocated: %12d bytes, %s\n",
		                     i->second.size,
		                     i->second.tag.c_str() );
				}
			}
            cudaDeviceReset();
            exit( 0 );
        }
        DevicePointerAttr attr;
        cudaPointerGetAttributes( &attr, ptr );
        attr.size   = nElem * sizeof( TYPE ) ;
        attr.pitch  = attr.size;
        attr.height = 1;
        attr.tag    = tag;
        mem_table[ ptr ] = attr ;
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> GPU[%d] + : %s\n", attr.device, size2text( attr.size ).c_str() );
#endif
        return ptr;
    }

    template <typename TYPE> TYPE *malloc_device_pitch( std::string tag, std::size_t &pitch, std::size_t width, std::size_t height )
    {
        TYPE *ptr;
        if( cudaMallocPitch( &ptr, &pitch, width * sizeof( TYPE ), height ) != cudaSuccess ) {
            fprintf( stderr, "<MESO> %s for %s, allocated = %s\n",
            		 cudaGetErrorString( cudaPeekAtLastError() ),
            		 tag.c_str(),
            		 size2text( device_allocated() ).c_str() );
            for( typename std::map<void *, DevicePointerAttr>::iterator i = mem_table.begin(); i != mem_table.end(); i++ ) {
				if ( i->second.memoryType == cudaMemoryTypeDevice ) {
		            fprintf( stderr, "<MESO> allocated: %12d bytes, %s\n",
		                     i->second.size,
		                     i->second.tag.c_str() );
				}
			}
            cudaDeviceReset();
            exit( 0 );
        }
        DevicePointerAttr attr;
        cudaPointerGetAttributes( &attr, ptr );
        attr.pitch  = pitch;
        attr.height = height;
        attr.size   = pitch * height;
        attr.tag    = tag;
        mem_table[ ptr ] = attr ;
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> GPU[%d] + : %s\n", attr.device, size2text( attr.size ).c_str() );
#endif
        return ptr;
    }

    template <typename TYPE> void realloc_device( std::string tag, TYPE*& ptr, std::size_t nElem, bool copy = true, bool zero = false )
    {
        std::map<void *, DevicePointerAttr >::iterator p;
        p = mem_table.find( ptr ) ;
        if( p == mem_table.end() ) { // if NULL pointer
            ptr = malloc_device<TYPE>( tag, nElem );
            if( zero ) cudaMemset( ptr, 0, nElem * sizeof( TYPE ) );
            return;
        } else if( p->second.memoryType == cudaMemoryTypeHost ) {
            fprintf( stderr, "<MESO> Host memory cannot be re-allocated to Device memory %p\n", ptr );
            cudaDeviceReset();
            exit( 0 );
        } else if( p->second.size == nElem * sizeof( TYPE ) ) return;

        if( copy ) {
            TYPE* ptr_new = malloc_device<TYPE>( tag, nElem );
            int   new_len = nElem * sizeof( TYPE );
            int   old_len = p->second.size;
            cudaMemcpy( ptr_new, ptr, min( old_len, new_len ), cudaMemcpyDefault );
            free( ptr );
            ptr = ptr_new;
        } else {
            free( ptr );
            ptr = malloc_device<TYPE>( tag, nElem );
            if( zero ) cudaMemset( ptr, 0, nElem * sizeof( TYPE ) );
        }
    }

    template <typename TYPE> void realloc_device_pitch( std::string tag, TYPE*& ptr, std::size_t &pitch, std::size_t width, std::size_t height, bool copy = true, bool zero = false )
    {
        std::map<void *, DevicePointerAttr >::iterator p;
        p = mem_table.find( ptr ) ;
        if( p == mem_table.end() ) { // if NULL pointer
            ptr = malloc_device_pitch<TYPE>( tag, pitch, width, height );
            if( zero ) cudaMemset( ptr, 0, pitch * height );
            return;
        } else if( p->second.memoryType == cudaMemoryTypeHost ) {
            fprintf( stderr, "<MESO> Host memory cannot be re-allocated to Device memory %p\n", ptr );
            cudaDeviceReset();
            exit( 0 );
        } else if( p->second.pitch == width * sizeof( TYPE ) && p->second.height == height ) return;

        if( copy ) {
            TYPE* ptr_new = malloc_device_pitch<TYPE>( tag, pitch, width, height );
            int   new_width = pitch;
            int   old_width = p->second.pitch;
            int   new_height = height;
            int   old_height = p->second.height;
            cudaMemcpy2D( ptr_new, pitch, ptr, p->second.pitch, min( old_width, new_width ), min( old_height, new_height ), cudaMemcpyDefault );
            free( ptr );
            ptr = ptr_new;
        } else {
            free( ptr );
            ptr = malloc_device_pitch<TYPE>( tag, pitch, width, height );
            if( zero ) cudaMemset( ptr, 0, pitch * height * sizeof( TYPE ) );
        }
    }

    template <typename TYPE> TYPE *malloc_host( std::string tag, std::size_t nElem )
    {
        TYPE *ptr;
        if( cudaMallocHost( &ptr, nElem * sizeof( TYPE ), cudaHostAllocMapped ) != cudaSuccess ) {
            fprintf( stderr, "<MESO> %s size = %s for %s\n",
                     cudaGetErrorString( cudaPeekAtLastError() ),
                     size2text( nElem * sizeof( TYPE ) ).c_str(),
                     tag.c_str() );
            cudaDeviceReset();
            exit( 0 );
        }
        DevicePointerAttr attr;
        cudaPointerGetAttributes( &attr, ptr );
        attr.size = nElem * sizeof( TYPE ) ;
        attr.pitch  = attr.size;
        attr.height = 1;
        attr.tag    = tag;
        mem_table[ ptr ] = attr ;
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> HOST + : %s\n", size2text( attr.size ).c_str() );
#endif
        return ptr;
    }

    template <typename TYPE> TYPE *malloc_host_pitch( std::string tag, std::size_t &pitch, std::size_t width, std::size_t height )
    {
        TYPE *ptr;
        pitch = ceiling( width * sizeof( TYPE ) + 127, 128 );
        if( cudaMallocHost( &ptr, pitch * height, cudaHostAllocMapped ) != cudaSuccess ) {
            fprintf( stderr, "<MESO> %s size = %s, allocated = %s\n",
                     cudaGetErrorString( cudaPeekAtLastError() ),
                     size2text( pitch * height ).c_str(),
                     size2text( host_allocated() ).c_str() );
            cudaDeviceReset();
            exit( 0 );
        }
        DevicePointerAttr attr;
        cudaPointerGetAttributes( &attr, ptr );
        attr.size   = pitch * height ;
        attr.pitch  = pitch;
        attr.height = height ;
        attr.tag    = tag;
        mem_table[ ptr ] = attr ;
#ifdef LMP_MESO_LOG_MEM
        fprintf( stderr, "<MESO> HOST + : %s\n", size2text( attr.size ).c_str() );
#endif
        return ptr;
    }

    template <typename TYPE> void realloc_host( std::string tag, TYPE*& ptr, std::size_t nElem, bool copy = true, bool zero = false )
    {
        std::map<void *, DevicePointerAttr >::iterator p;
        p = mem_table.find( ptr ) ;
        if( p == mem_table.end() ) { // if NULL pointer
            ptr = malloc_host<TYPE>( tag, nElem );
            if( zero ) memset( ptr, 0, nElem * sizeof( TYPE ) );
            return;
        } else if( p->second.memoryType == cudaMemoryTypeDevice ) {
            fprintf( stderr, "<MESO> Device memory cannot be re-allocated to Host memory %p\n", ptr );
            cudaDeviceReset();
            exit( 0 );
        } else if( p->second.size == nElem * sizeof( TYPE ) ) return;

        if( copy ) {
            TYPE* ptr_new = malloc_host<TYPE>( tag, nElem );
            int   new_len = nElem * sizeof( TYPE );
            int   old_len = p->second.size;
            memcpy( ptr_new, ptr, min( old_len, new_len ) );
            free( ptr );
            ptr = ptr_new;
        } else {
            free( ptr );
            ptr = malloc_host<TYPE>( tag, nElem );
            if( zero ) memset( ptr, 0, nElem * sizeof( TYPE ) );
        }
    }

    inline double get_time_posix() const
    {
        struct timespec time;
        clock_gettime( CLOCK_REALTIME, &time );
        return ( double )time.tv_sec + ( double )time.tv_nsec * 1.0e-9 ;
    }
    inline double get_time_mpi() const
    {
        return MPI_Wtime();
    }
    inline double get_time_omp() const
    {
        return omp_get_wtime();
    }
};

}

#endif
