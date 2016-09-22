#ifndef LMP_MESO_ENGINE_UTIL
#define LMP_MESO_ENGINE_UTIL

#include "type_meso.h"
#include "util_meso.h"

namespace LAMMPS_NS
{

class CUDAEvent
{
public:
    CUDAEvent( const bool create = false ) : _e( NULL )
    {
        if( create ) watch( ( cudaEventCreateWithFlags( &_e, cudaEventDefault ) ) );
    }
    CUDAEvent( const CUDAEvent &another ): _e( NULL )
    {
        _e = another._e;
    }

    inline cudaEvent_t e() const
    {
        return _e;
    }
    inline operator cudaEvent_t () const
    {
        return e();
    }

    inline void record( const cudaStream_t stream ) const
    {
        watch( ( cudaEventRecord( e(), stream ) ) );
    }
    inline void sync() const
    {
        watch( ( cudaEventSynchronize( e() ) ) );
    }
    inline friend float operator - ( CUDAEvent end, CUDAEvent start )
    {
        float ms;
        watch( ( cudaEventElapsedTime( &ms, start, end ) ) );
        return ms * 1E-3;
    }
private:
    cudaEvent_t _e;
};

//class CUDAEventTagged
//{
//public:
//  CUDAEvent() : _tag(), _e(NULL) {}
//  CUDAEvent( const string tag ): _tag(tag), _e(NULL) {
//      watch(( cudaEventCreateWithFlags( &_e, cudaEventDefault ) ));
//  }
//  CUDAEvent( const CUDAEvent &another ): _tag(""), _e(NULL) {
//      *this = another;
//  }
//  inline CUDAEvent& operator = ( const CUDAEvent &another ) {
//      _e = another._e;
//      _tag = another._tag;
//      return *this;
//  }
//
//  inline cudaEvent_t e() const { return _e; }
//  inline operator cudaEvent_t () const { return e(); }
//
//  inline void record( const cudaStream_t stream ) const {
//      watch(( cudaEventRecord( e(), stream ) ));
//  }
//  inline void sync() const {
//      watch(( cudaEventSynchronize( e() ) ));
//  }
//private:
//  string _tag;
//  cudaEvent_t _e;
//};

struct CUDAStream {
public:
    CUDAStream( const bool create = false ): _s( NULL )
    {
        // watch(( cudaStreamCreateWithFlags( &_s, cudaStreamNonBlocking ) )); // do not use non blocking streams, not compatible with data transfer
        watch( ( cudaStreamCreate( &_s ) ) );
    }
    CUDAStream( const CUDAStream &another ): _s( NULL )
    {
        _s = another._s;
    }

    inline cudaStream_t s() const
    {
        return _s;
    }
    inline operator cudaStream_t () const
    {
        return s();
    }

    inline void sync() const
    {
        watch( ( cudaStreamSynchronize( s() ) ) );
    }
    inline void waiton( cudaEvent_t e ) const
    {
        watch( ( cudaStreamWaitEvent( _s, e, 0 ) ) );
    }
    inline static void all_waiton( cudaEvent_t e )
    {
        watch( ( cudaStreamWaitEvent( NULL, e, 0 ) ) );
    }
protected:
    cudaStream_t _s;
};

//class CUDAStreamTagged
//{
//public:
//  CUDAStream(): _tag(), _s(NULL) {}
//  CUDAStream( const string tag ): _tag(tag), _s(NULL) {
//      watch(( cudaStreamCreate( &_s ) ));
//  }
//  CUDAStream( const CUDAStream &another ): _tag(""), _s(NULL) {
//      *this = another;
//  }
//  inline CUDAStream& operator = ( const CUDAStream &another ) {
//      _s = another._s;
//      _tag = another._tag;
//      return *this;
//  }
//
//  inline cudaStream_t s() const { return _s; }
//  inline operator cudaStream_t () const { return s(); }
//
//  inline void sync() const {
//      watch(( cudaStreamSynchronize( s() ) ));
//  }
//  inline void waiton( cudaEvent_t e ) const {
//      watch(( cudaStreamWaitEvent( _s, e, 0 ) ));
//  }
//  inline void waiton( CUDAEvent e ) const {
//      waiton( e.e() );
//  }
//  inline static void all_waiton( cudaEvent_t e ) {
//      watch(( cudaStreamWaitEvent( NULL, e, 0 ) ));
//  }
//  inline static void all_waiton( CUDAEvent e ) {
//      all_waiton( e.e() );
//  }
//private:
//  string _tag;
//  cudaStream_t _s;
//};

struct GridConfig: public int2 {
    GridConfig() {}
    GridConfig( const int2 baseline_ ) : int2( baseline_ ) {}
    GridConfig( const GridConfig &other ) : int2( other ) {}

    int partition( const int parallelism, const int max_partition, const double penalty = 0.0 )
    {
        if( parallelism < x * y && parallelism > 0 ) {
            int best_n_partition = 0;
            double best_vacancy = 1.0 + max_partition * penalty;
            for( int n_partition = 1; n_partition <= min( x, max_partition ); n_partition++ ) {
                int block_per_part = x / n_partition;
                int threads_per_part = y * block_per_part;
                double parallelism_per_thread = double( parallelism ) / double( threads_per_part );
                double thread_occupancy = parallelism_per_thread / ceil( parallelism_per_thread );
                double block_occupancy = double( block_per_part * n_partition ) / double( x );
                double occupancy = block_occupancy * thread_occupancy;
                double vacancy = 1.0 - occupancy;
                //printf("\tn_partition %d\tsize %d\tparallelism_per_thread %.2lf\tthread_occupancy %.2lf\tblock_occupancy %.2lf\toccupancy %.2lf\twaste: %.2lf=(%.2lf+%.2lf), %.2lf=(%.2lf+%.2lf)\n", n_partition, threads_per_part, parallelism_per_thread, thread_occupancy, block_occupancy, occupancy, vacancy + n_partition * penalty, vacancy, n_partition * penalty, best_vacancy + best_n_partition * penalty, best_vacancy, best_n_partition * penalty );
                if( vacancy + n_partition * penalty < best_vacancy + best_n_partition * penalty ) {
                    best_vacancy = vacancy;
                    best_n_partition = n_partition;
                }
            }
            //printf("best #partition for %d X %d threads vs. %d work: %d\n", x, y, parallelism, best_n_partition );
            return best_n_partition;
        } else {
//          printf("best #partition for %d X %d threads vs. %d work: %d\n", x, y, parallelism, 1 );
            return 1;
        }
    }

    int2 dynamic( const int parallelism, const int min_block_size = 32, const int max_block_size = 1024 ) const
    {
        if( parallelism < x * y ) {  // unsaturated
            int2 divide;
            divide.x = x;
            divide.y = min( max( ceiling( parallelism / divide.x, 32 ), min_block_size ), max_block_size );
            divide.x = ceiling( parallelism, divide.y ) / divide.y;
            //printf("DYNAMIC: %d %d, STATIC: %d %d, total: %d %d)\n",divide.x,divide.y, x, y, divide.x*divide.y, parallelism);
            return divide;
        } else {
            return make_int2( x, y );
        }
    }
};

}

#endif
