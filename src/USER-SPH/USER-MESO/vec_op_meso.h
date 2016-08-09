#ifndef LMP_MESO_VECTOR_OPERATOR
#define LMP_MESO_VECTOR_OPERATOR
#ifdef __CUDACC__

#define MAKE_VOP(VTYPE,STYPE) \
__inline__ __host__ __device__ VTYPE operator + ( VTYPE const & a, VTYPE const & b ) { \
    return make_##VTYPE( a.x + b.x, a.y + b.y, a.z + b.z ); \
} \
__inline__ __host__ __device__ VTYPE operator - ( VTYPE const & a, VTYPE const & b ) { \
    return make_##VTYPE( a.x - b.x, a.y - b.y, a.z - b.z ); \
} \
__inline__ __host__ __device__ VTYPE operator - ( VTYPE const & a ) { \
    return make_##VTYPE( -a.x, -a.y, -a.z ); \
} \
__inline__ __host__ __device__ VTYPE operator * ( VTYPE const & a, VTYPE const & b ) { \
    return make_##VTYPE( a.x * b.x, a.y * b.y, a.z * b.z ); \
} \
__inline__ __host__ __device__ VTYPE operator * ( VTYPE const & a, STYPE b ) { \
    return make_##VTYPE( a.x * b, a.y * b, a.z * b ); \
} \
__inline__ __host__ __device__ VTYPE operator * ( STYPE b, VTYPE const & a ) { \
    return make_##VTYPE( a.x * b, a.y * b, a.z * b ); \
} \
__inline__ __host__ __device__ VTYPE operator / ( VTYPE const & a, STYPE b ) { \
    return make_##VTYPE( a.x * (1. / b), a.y * (1. / b), a.z * (1. / b) ); \
} \
__inline__ __host__ __device__ STYPE dot( VTYPE const & a, VTYPE const & b ) { \
    return a.x * b.x + a.y * b.y + a.z * b.z; \
} \
__inline__ __host__ __device__ STYPE normsq( VTYPE const & a ) { \
    return dot( a, a );\
} \
__inline__ __host__ __device__ STYPE norm( VTYPE const & a ) { \
     return sqrt( normsq(a) );\
 } \
__inline__ __host__ __device__ VTYPE normalize( VTYPE const & a ) { \
     return a * rsqrt( normsq(a) ); \
 } \
__inline__ __host__ __device__ STYPE reduce_sum( VTYPE const & a ) { \
    return a.x + a.y + a.z; \
} \
__inline__ __host__ __device__ STYPE reduce_mul( VTYPE const & a ) { \
    return a.x * a.y * a.z; \
} \
__inline__ __host__ __device__ STYPE reduce_max( VTYPE const & a ) { \
    return max( max( a.x, a.y ), a.z ); \
} \
__inline__ __host__ __device__ STYPE reduce_min( VTYPE const & a ) { \
    return min( min( a.x, a.y ), a.z ); \
} \


MAKE_VOP(double3,double)
MAKE_VOP(float3,float)

#undef MAKE_VOP

#endif
#endif
