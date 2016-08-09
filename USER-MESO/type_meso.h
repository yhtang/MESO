#ifndef LMP_MESO_TYPE
#define LMP_MESO_TYPE

#include "util_meso.h"
#include "math_meso.h"

namespace LAMMPS_NS
{

typedef double       r64;
typedef float        r32;
typedef unsigned long long u64;
typedef cudaTextureObject_t texobj;
typedef void( *void_fptr )();

typedef struct float3uint {
    r32 x, y, z;
    uint i;
    __inline__ __device__ float3uint( const float4& v )
    {
        *this = v;
    }
    __inline__ __device__ float3uint& operator = ( const float4& v )
    {
        x = v.x;
        y = v.y;
        z = v.z;
        i = __float_as_uint( v.w );
        return *this;
    }
} f3u;

typedef struct float3int {
    r32 x, y, z;
    int i;
    __inline__ __device__ float3int( const float4& v )
    {
        *this = v;
    }
    __inline__ __device__ float3int& operator = ( const float4& v )
    {
        x = v.x;
        y = v.y;
        z = v.z;
        i = __float_as_int( v.w );
        return *this;
    }
} f3i;

typedef struct temp_massinv_misc {
    r32 T, mass_inv, unused1;
    uint i;
    __inline__ __device__ temp_massinv_misc( const float4& v )
    {
        *this = v;
    }
    __inline__ __device__ temp_massinv_misc& operator = ( const float4& v )
    {
        T = v.x;
        mass_inv = v.y;
        unused1 = v.z;
        i = __float_as_uint( v.w );
        return *this;
    }
} tmm;

}

#endif
