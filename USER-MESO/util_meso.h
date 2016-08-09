#ifndef LMP_MESO_UTIL
#define LMP_MESO_UTIL

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <csignal>
#include <queue>
#include <limits>

#include "omp.h"
#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "lib_dispatcher_meso.h"

namespace LAMMPS_NS
{

// hardware specs
const static unsigned int CPU_CACHELINE_WORD = 16;
const static int OMPDEBUG = 0;

//#define LMP_MESO_LOG_MEM
//#define LMP_MESO_LOG_L1
//#define LMP_MESO_LOG_L2
//#define LMP_MESO_LOG_L3
#define LMP_MESO_STRICT 0

struct AtomAttribute {
    typedef unsigned long long Descriptor;
    static const unsigned long long BEGIN = 1ULL;
    static const unsigned long long FIELD_MASK = ( 1ULL << 48 ) - 1ULL;
    static const unsigned long long RANGE_MASK = ~FIELD_MASK;

    static const unsigned long long NONE  = 0ULL;
    static const unsigned long long TYPE  = 1ULL <<  0;
    static const unsigned long long MASS  = 1ULL <<  1;
    static const unsigned long long MASK  = 1ULL <<  2;
    static const unsigned long long TAG   = 1ULL <<  3;
    static const unsigned long long COORD = 1ULL <<  4;
    static const unsigned long long FORCE = 1ULL <<  5;
    static const unsigned long long VELOC = 1ULL <<  6;
    static const unsigned long long TEMP  = 1ULL <<  7;
    static const unsigned long long HEAT  = 1ULL <<  8;
    static const unsigned long long RHO   = 1ULL <<  9;
    static const unsigned long long IMAGE = 1ULL << 10;
    static const unsigned long long MOLE  = 1ULL << 11;
    static const unsigned long long EXCL  = 1ULL << 12;
    static const unsigned long long BOND  = 1ULL << 13;
    static const unsigned long long ANGLE = 1ULL << 14;
    static const unsigned long long DIHED = 1ULL << 15;
    static const unsigned long long ESSENTIAL = TYPE | TAG | MASS | MASK | COORD | VELOC | IMAGE;
    // to be continued...

    static const unsigned long long BULK   = 1ULL << 61;
    static const unsigned long long BORDER = 1ULL << 62;
    static const unsigned long long GHOST  = 1ULL << 63;
    static const unsigned long long LOCAL  = BULK  | BORDER;
    static const unsigned long long ALL    = LOCAL | GHOST;
};

#define ACTION_COPY         1
#define ACTION_PERM         2

enum TransferDirection {
    CUDACOPY_C2G, CUDACOPY_G2C
};

// all: from init to finalize
// loop: from integrate::setup to last step of integrate::iterate
// core: from the 25% - 75% steps
enum {
    CUDAPROF_NONE       = 0,
    CUDAPROF_ALL        = 1,
    CUDAPROF_LOOP       = 2,
    CUDAPROF_CORE       = 3,
    CUDAPROF_INTVL      = 4
};

//#define verify(cuapi) {cudaError_t err=(cuapi);if(err)error->one(__FILE__,__LINE__,cudaGetErrorString(err));}
//#define watch(cuapi)  {cudaError_t err=(cuapi);if(err)fprintf(stderr,"%s %s %s\n",__FILE__,__LINE__,cudaGetErrorString(err));}
#define verify(cuapi)   {cudaError_t err=(cuapi);if(err){fprintf(stderr,"%s\n",cudaGetErrorString(err));raise(SIGABRT);}}
#define watch(cuapi)    {cudaError_t err=(cuapi);if(err)raise(SIGABRT);}
#define fast_exit(e)    cudaDeviceSynchronize();\
		                std::cout<<cudaGetErrorString(cudaGetLastError())<<' '<<__LINE__<<' '<<__FILE__<<std::endl;\
                        MPI_Finalize();\
                        exit(e);
#define fast_debug()    cudaDeviceSynchronize();\
                        std::cout<<cudaGetErrorString(cudaGetLastError())<<' '<<__LINE__<<' '<<__FILE__<<std::endl;

#ifdef __CUDACC__
template<bool P> class STATIC_CHECKER
{
public:
    __device__ STATIC_CHECKER( bool p ) {}
};
template<> class STATIC_CHECKER<false>;
#define device_static_assert(PRED,MSG) \
    {STATIC_CHECKER<PRED> ERROR_##MSG(PRED);}
#endif

template<typename T>
__global__ void check_acc( T *x, T *y, T *z, const int n ) {
	double ax = 0, ay = 0, az = 0;
	for(int i=0;i<n;i++) {
		ax += x[i];
		ay += y[i];
		az += z[i];
	}
	printf("net force: %lf %lf %lf\n", ax, ay, az );
}

template<typename T, typename S> inline void pack( double *r, T x, S y )
{
    T *p = ( T* )r;
    *p++ = x;
    *( S* )p = y;
}
template<typename T, typename S> inline void unpack( double *r, T &x, S &y )
{
    T *p = ( T* )r;
    x = *p++;
    y = *( S* )p;
}
template<typename T> inline void pack( double *r, T x, T y )
{
    T *p = ( T* )r;
    *p++ = x;
    *p++ = y;
}
template<typename T> inline void unpack( double *r, T &x, T &y )
{
    T *p = ( T* )r;
    x = *p++;
    y = *p++;
}
template<typename T> inline void pack( double *r, T x )
{
    *( ( T* )r ) = x;
}
template<typename T> inline void unpack( double *r, T &x )
{
    x = *( ( T* )r );
}

inline int2 split_work( long long beg, long long end, long long tid, long long ntd )
{
    long long n = end - beg;
    return make_int2( beg + tid * n / ntd, beg + ( ( tid == ntd - 1 ) ? n : ( tid + 1 ) * n / ntd ) );
}

//// IO operators for CUDA vector types
//template<typename TYPE>
//std::string serialize_1 ( const TYPE v ) {
//  std::stringstream str;
//  str<<'('<<v.x<<')';
//  return str.str();
//}
//
//template<typename TYPE>
//std::string serialize_2 ( const TYPE v ) {
//  std::stringstream str;
//  str<<'('<<v.x<<','<<v.y<<')';
//  return str.str();
//}
//
//template<typename TYPE>
//std::string serialize_3 ( const TYPE v ) {
//  std::stringstream str;
//  str<<'('<<v.x<<','<<v.y<<','<<v.z<<')';
//  return str.str();
//}
//
//template<typename TYPE>
//std::string serialize_4 ( const TYPE v ) {
//  std::stringstream str;
//  str<<'('<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<')';
//  return str.str();
//}
//
//#define SERIALIZER_1(type) \
//  inline std::ostream& operator << ( std::ostream &out, const type v ) { \
//      return out.operator<<( serialize_1(v).c_str() ); \
//  }
//
//SERIALIZER_1(char1)
//SERIALIZER_1(double1)
//SERIALIZER_1(float1)
//SERIALIZER_1(int1)
//SERIALIZER_1(long1)
//SERIALIZER_1(longlong1)
//SERIALIZER_1(short1)
//SERIALIZER_1(uchar1)
//SERIALIZER_1(uint1)
//SERIALIZER_1(ulong1)
//SERIALIZER_1(ulonglong1)
//SERIALIZER_1(ushort1)
//
//#undef SERIALIZER_1
//
//#define SERIALIZER_2(type) \
//  inline std::ostream& operator << ( std::ostream &out, const type v ) { \
//      return out.operator<<( serialize_2(v).c_str() ); \
//  }
//
//SERIALIZER_2(char2)
//SERIALIZER_2(double2)
//SERIALIZER_2(float2)
//SERIALIZER_2(int2)
//SERIALIZER_2(long2)
//SERIALIZER_2(longlong2)
//SERIALIZER_2(short2)
//SERIALIZER_2(uchar2)
//SERIALIZER_2(uint2)
//SERIALIZER_2(ulong2)
//SERIALIZER_2(ulonglong2)
//SERIALIZER_2(ushort2)
//
//#undef SERIALIZER_2
//
//#define SERIALIZER_3(type) \
//  inline std::ostream& operator << ( std::ostream &out, const type v ) { \
//      return out.operator<<( serialize_3(v).c_str() ); \
//  }
//
//SERIALIZER_3(char3)
//SERIALIZER_3(float3)
//SERIALIZER_3(int3)
//SERIALIZER_3(long3)
//SERIALIZER_3(short3)
//SERIALIZER_3(uchar3)
//SERIALIZER_3(uint3)
//SERIALIZER_3(ulong3)
//SERIALIZER_3(ushort3)
//
//#undef SERIALIZER_3
//
//#define SERIALIZER_4(type) \
//  inline std::ostream& operator << ( std::ostream &out, const type v ) { \
//      out<<( serialize_4(v).c_str() ); \
//      return out; \
//  }
//
//SERIALIZER_4(char4)
//SERIALIZER_4(float4)
//SERIALIZER_4(int4)
//SERIALIZER_4(long4)
//SERIALIZER_4(short4)
//SERIALIZER_4(uchar4)
//SERIALIZER_4(uint4)
//SERIALIZER_4(ulong4)
//SERIALIZER_4(ushort4)
//
//#undef SERIALIZER_4

}

#endif
