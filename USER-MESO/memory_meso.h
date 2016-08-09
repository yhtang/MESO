#ifndef LMP_MESO_MEMOBJ
#define LMP_MESO_MEMOBJ

#include <vector>
#include "error.h"
#include "engine_meso.h"

namespace LAMMPS_NS
{

template<typename TYPE> __global__ void gpu_buffer_set( TYPE* buffer, const TYPE v, const size_t n )
{
    for( size_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < n ; i += gridDim.x * blockDim.x )
        buffer[i] = v;
}

template<typename TYPE> __global__ void gpu_buffer_vset( TYPE** buffer, const TYPE v, const size_t offset, const size_t n, const uint D )
{
    for( size_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < n ; i += gridDim.x * blockDim.x ) {
        for( uint d = 0 ; d < D ; d++ ) buffer[d][offset + i] = v;
    }
}

template<typename TYPE> class DeviceScalar: protected Pointers, protected MesoPointers
{
public:
    DeviceScalar( LAMMPS *lmp, std::string tag, size_t n = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _n( 0 )
    {
        _tag = tag;
        if( n ) grow( n );
    }
    DeviceScalar( const DeviceScalar<TYPE> &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _n( 0 )
    {
        *this = another;
    }
    inline DeviceScalar<TYPE>& operator = ( const DeviceScalar<TYPE> &another )
    {
        destroy();
        _tag = another._tag;
        if( another._n ) grow( another._n );
        return *this;
    }
    ~DeviceScalar()
    {
        destroy();
    }
    inline operator TYPE* () const
    {
        return ptr();
    }

    inline TYPE* ptr() const
    {
        return _ptr;
    }
    inline size_t n() const
    {
        return _n;
    }
    inline size_t size() const
    {
        return n() * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    inline DeviceScalar<TYPE>* grow( size_t n, bool copy = true, bool zero = false )
    {
        if( n ) {
            if( !_ptr ) {
            	_ptr = meso_device->malloc_device<TYPE>( _tag, n );
            	if (zero) verify( ( cudaMemset( _ptr, 0, n * sizeof(TYPE) ) ) );
            }
            else meso_device->realloc_device( _tag, _ptr, n, copy, zero );
            _n = n;
        } else {
            destroy();
        }
        return this;
    }
    inline void upload( TYPE *ptr_h, size_t n, cudaStream_t stream = 0, size_t offset = 0 )
    {
        if( _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %lu", tag().c_str(), _ptr, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( _ptr + offset, ptr_h, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( _ptr + offset, ptr_h, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }
    inline void download( TYPE *ptr_h, size_t n, cudaStream_t stream = 0, size_t offset = 0 ) const
    {
        if( _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %lu", tag().c_str(), _ptr, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( ptr_h, _ptr + offset, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( ptr_h, _ptr + offset, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }
    inline void set( TYPE v, cudaStream_t stream = 0, size_t offset = 0, size_t count = SIZE_MAX )
    {
        if( count == SIZE_MAX ) count = _n - offset;
        if( offset + count > _n ) {
            char info[512];
            sprintf( info, "<MESO> %s device buffer size %lu smaller than requested %lu", tag().c_str(), _n, offset + count );
            error->one( FLERR, info );
        }
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_buffer_set<TYPE>, 0 );
        if( stream != 0 ) {
            gpu_buffer_set <<< grid_cfg.x, grid_cfg.y, 0, stream >>>( _ptr + offset, v, count );
            verify( ( cudaPeekAtLastError() ) );
        } else {
            gpu_buffer_set <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> ( _ptr + offset, v, count );
            verify( ( cudaStreamSynchronize( meso_device->stream() ) ) );
        }
    }
protected:
    inline void destroy()
    {
        if( _ptr ) meso_device->free( _ptr );
        _ptr = NULL;
        _n = 0;
    }

    std::string _tag;
    TYPE* _ptr;
    size_t _n;
};

template<typename TYPE> class HostScalar: protected Pointers, protected MesoPointers
{
public:
    HostScalar( LAMMPS *lmp, std::string tag, size_t n = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _n( 0 )
    {
        _tag = tag;
        if( n ) grow( n );
    }
    HostScalar( const HostScalar<TYPE> &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _n( 0 )
    {
        *this = another;
    }
    inline const HostScalar<TYPE>& operator = ( const HostScalar<TYPE> &another )
    {
        destroy();
        _tag = another._tag;
        if( another._n ) grow( another._n );
        return *this;
    }
    ~HostScalar()
    {
        destroy();
    }
    inline operator TYPE* () const
    {
        return ptr();
    }

    inline TYPE* ptr() const
    {
        return _ptr;
    }
    inline size_t n() const
    {
        return _n;
    }
    inline size_t size() const
    {
        return n() * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    inline HostScalar<TYPE>* grow( size_t n, bool copy = true, bool zero = false )
    {
        if( n ) {
            if( !_ptr ) _ptr = meso_device->malloc_host<TYPE>( _tag, n );
            else meso_device->realloc_host( _tag, _ptr, n, copy, zero );
            _n = n;
        } else {
            destroy();
        }
        return this;
    }
    inline void copy( std::vector<TYPE> &vec )
    {
        if( vec.size() > n() ) raise( SIGSEGV );
        for( int i = 0 ; i < vec.size() ; i++ ) _ptr[i] = vec[i];
    }

protected:
    inline void destroy()
    {
        if( _ptr ) meso_device->free( _ptr );
        _ptr = NULL;
        _n = 0;
    }

    std::string _tag;
    TYPE* _ptr;
    size_t _n;
};

#if !defined(__arm__) && !defined(__aarch64__)
template<typename TYPE> class Pinned: protected Pointers, protected MesoPointers
{
public:
    Pinned( LAMMPS *lmp, std::string tag, size_t n = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( "" ),
        _buf_d( NULL ),
        _ptr_d( NULL ),
        _ptr_h( NULL ),
        _n( 0 )
    {
        _tag = tag;
    }
    Pinned( const DeviceScalar<TYPE> &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ),
        _buf_d( NULL ),
        _ptr_d( NULL ),
        _ptr_h( NULL ),
        _n( 0 )
    {
        *this = another;
    }
    inline const Pinned<TYPE>& operator = ( const Pinned<TYPE> &another )
    {
        destroy();
        _tag = another._tag;
        if( another._ptr_h ) map_host( another._n, another._ptr_h );
        return *this;
    }
    ~Pinned()
    {
        destroy();
    }

    inline TYPE* buf_d() const
    {
        return _buf_d;    // address of device buffer
    }
    inline TYPE* ptr_d() const
    {
        return _ptr_d;    // address of host buffer on the device space
    }
    inline TYPE* ptr_h() const
    {
        return _ptr_h;    // address of host buffer
    }
    inline size_t n() const
    {
        return _n;
    }
    inline size_t size() const
    {
        return n() * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    inline void map_host( size_t n, TYPE* ptr )
    {
        if( !n ) {
            char info[512];
            sprintf( info, "<MESO> memory object %s mapping size is 0", tag().c_str() );
            error->one( FLERR, info );
            raise( SIGSEGV );
            return;
        }
        if( _ptr_h ) {
            char info[512];
            sprintf( info, "<MESO> memory object %s already mapped to %p", tag().c_str(), ptr_h() );
            error->warning( FLERR, info );
        } else {
            if( _n != n ) resize( n, false, false );
            _ptr_h = ptr;
            verify( ( cudaHostRegister( _ptr_h, _n * sizeof( TYPE ), cudaHostRegisterMapped ) ) );
            verify( ( cudaHostGetDevicePointer( &_ptr_d, _ptr_h, 0 ) ) );
        }
    }
    inline void unmap_host( TYPE* ptr_h = NULL )
    {
        if( !_ptr_h ) return;
        if( _ptr_h != ptr_h ) {
            char info[512];
            sprintf( info, "<MESO> host buffer address for %s changed since last pinning", tag().c_str() );
            error->warning( FLERR, info );
        }
        verify( ( cudaHostUnregister( _ptr_h ) ) );
        _ptr_h = NULL;
        _ptr_d = NULL;
    }
    inline void upload( size_t n, cudaStream_t stream = 0, size_t offset = 0 )
    {
        if( !_ptr_h || _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %d", tag().c_str(), _ptr_h, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( _buf_d + offset, _ptr_h + offset, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( _buf_d + offset, _ptr_h + offset, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }
    inline void download( size_t n, cudaStream_t stream = 0, size_t offset = 0 ) const
    {
        if( !_ptr_h || _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %d", tag().c_str(), _ptr_h, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( _ptr_h + offset, _buf_d + offset, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( _ptr_h + offset, _buf_d + offset, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }

protected:
    inline void resize( size_t n, bool copy = true, bool zero = false )
    {
        if( n ) {
            if( !_buf_d ) _buf_d = meso_device->malloc_device<TYPE>( _tag, n );
            else meso_device->realloc_device( _tag, _buf_d, n, copy, zero );
            _n = n;
        } else {
            destroy();
        }
    }
    inline void destroy()
    {
        if( _ptr_h ) unmap_host( _ptr_h );
        if( _buf_d ) meso_device->free( _buf_d );
        _ptr_h = NULL;
        _buf_d = NULL;
        _n = 0;
    }

    std::string _tag;
    TYPE* _buf_d;
    TYPE* _ptr_d;
    TYPE* _ptr_h;
    size_t _n;
};
#else
template<typename TYPE> class Pinned: protected Pointers, protected MesoPointers
{
public:
    Pinned( LAMMPS *lmp, std::string tag, size_t n = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( "" ),
        _buf_d( NULL ),
        _ptr_h( NULL ),
        _n( 0 )
    {
        _tag = tag;
    }
    Pinned( const DeviceScalar<TYPE> &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ),
        _buf_d( NULL ),
        _ptr_h( NULL ),
        _n( 0 )
    {
        *this = another;
    }
    inline const Pinned<TYPE>& operator = ( const Pinned<TYPE> &another )
    {
        destroy();
        _tag = another._tag;
        if( another._ptr_h ) map_host( another._n, another._ptr_h );
        return *this;
    }
    ~Pinned()
    {
        destroy();
    }

    inline TYPE* buf_d() const
    {
        return _buf_d;    // address of device buffer
    }
    inline TYPE* ptr_h() const
    {
        return _ptr_h;    // address of host buffer
    }
    inline size_t n() const
    {
        return _n;
    }
    inline size_t size() const
    {
        return n() * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    inline void map_host( size_t n, TYPE* ptr )
    {
        if( !n ) {
            char info[512];
            sprintf( info, "<MESO> memory object %s mapping size is 0", tag().c_str() );
            error->one( FLERR, info );
            raise( SIGSEGV );
            return;
        }
        if( _ptr_h ) {
            char info[512];
            sprintf( info, "<MESO> memory object %s already mapped to %p", tag().c_str(), ptr_h() );
            error->warning( FLERR, info );
        } else {
            if( _n != n ) resize( n, false, false );
            _ptr_h = ptr;
        }
    }
    inline void unmap_host( TYPE* ptr_h = NULL )
    {
        if( !_ptr_h ) return;
        if( _ptr_h != ptr_h ) {
            char info[512];
            sprintf( info, "<MESO> host buffer address for %s changed since last pinning", tag().c_str() );
            error->warning( FLERR, info );
        }
        _ptr_h = NULL;
    }
    inline void upload( size_t n, cudaStream_t stream = 0, size_t offset = 0 )
    {
        if( !_ptr_h || _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %d", tag().c_str(), _ptr_h, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( _buf_d + offset, _ptr_h + offset, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( _buf_d + offset, _ptr_h + offset, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }
    inline void download( size_t n, cudaStream_t stream = 0, size_t offset = 0 ) const
    {
        if( !_ptr_h || _n < n + offset ) {
            char info[512];
            sprintf( info, "<MESO> %s host buffer address %p not allocated or smaller than request size: %d", tag().c_str(), _ptr_h, _n );
            error->one( FLERR, info );
        }
        if( stream != 0 ) {
            verify( ( cudaMemcpyAsync( _ptr_h + offset, _buf_d + offset, n * sizeof( TYPE ), cudaMemcpyDefault, stream ) ) );
        } else {
            verify( ( cudaMemcpy( _ptr_h + offset, _buf_d + offset, n * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        }
    }

protected:
    inline void resize( size_t n, bool copy = true, bool zero = false )
    {
        if( n ) {
            if( !_buf_d ) _buf_d = meso_device->malloc_device<TYPE>( _tag, n );
            else meso_device->realloc_device( _tag, _buf_d, n, copy, zero );
            _n = n;
        } else {
            destroy();
        }
    }
    inline void destroy()
    {
        if( _ptr_h ) unmap_host( _ptr_h );
        if( _buf_d ) meso_device->free( _buf_d );
        _ptr_h = NULL;
        _buf_d = NULL;
        _n = 0;
    }

    std::string _tag;
    TYPE* _buf_d;
    TYPE* _ptr_h;
    size_t _n;
};
#endif

template<typename TYPE> class DevicePitched: protected Pointers, protected MesoPointers
{
public:
    DevicePitched( LAMMPS *lmp, std::string tag, size_t w = 0, size_t h = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _pitch( 0 ),
        _h( 0 )
    {
        _tag = tag;
        if( w && h ) grow( w, h );
    }
    DevicePitched( const DeviceScalar<TYPE> &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ),
        _ptr( NULL ),
        _pitch( 0 ),
        _h( 0 )
    {
        *this = another;
    }
    inline const DevicePitched<TYPE>& operator = ( const DevicePitched<TYPE> &another )
    {
        destroy();
        _tag = another._tag;
        if( another._pitch && another._h ) grow( another._pitch, another._h );
        return *this;
    }
    ~DevicePitched()
    {
        destroy();
    }
    inline operator TYPE* () const
    {
        return ptr();
    }

    inline TYPE* ptr() const
    {
        return _ptr;
    }
    inline size_t pitch() const
    {
        return _pitch;
    }
    inline size_t h() const
    {
        return _h;
    }
    inline size_t size() const
    {
        return h() * pitch() * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    inline DevicePitched<TYPE>* grow( size_t w, size_t h, bool copy = true, bool zero = false )
    {
        if( w && h ) {
            if( !_ptr ) _ptr = meso_device->malloc_device_pitch<TYPE>( _tag, _pitch, w, h );
            else meso_device->realloc_device_pitch( _tag, _ptr, _pitch, w, h, copy, zero );
            _pitch /= sizeof( TYPE );
            _h = h;
        } else {
            destroy();
        }
        return this;
    }

    template<bool TRANSPOSE>
    void dump( const char filename[] ) // for debugging
    {
        std::vector<TYPE> v( pitch() * h() );
        std::ofstream fout( filename );
        verify( ( cudaMemcpy( v.data(), ptr(), v.size() * sizeof( TYPE ), cudaMemcpyDefault ) ) );
        for( int i = 0 ; i < ( TRANSPOSE ? pitch() : h() ) ; i++ ) {
            for( int j = 0 ; j < ( TRANSPOSE ? h() : pitch() ) ; j++ ) {
                fout << v[ TRANSPOSE ? ( i + j * pitch() ) : ( i * pitch() + j ) ] << '\t';
            }
            fout << std::endl;
        }
    }

protected:
    inline void destroy()
    {
        if( _ptr ) meso_device->free( _ptr );
        _ptr = NULL;
        _pitch = 0;
    }

    std::string _tag;
    TYPE* _ptr;
    size_t _pitch, _h;
};

enum {
	_DYNAMIC_ = UINT_MAX
};

template<typename TYPE, uint D = _DYNAMIC_> class DeviceVector: protected Pointers, protected MesoPointers
{
public:
    DeviceVector( LAMMPS *lmp, std::string tag, size_t n = 0, uint d = 0 ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _d( D != _DYNAMIC_ ? D : d ),
        _ptr( NULL ),
        _ptrs( lmp, tag + "::ptrs", _d ),
        _tag( tag )
    {
        grow( n );
    }
    ~DeviceVector()
    {
        destroy();
    }

    inline TYPE* operator []( uint i ) const
    {
        return _ptr + i * _pitch;
    }
    inline size_t d() const
    {
        return _d;
    }
    inline void set_d(uint d)
    {
    	if ( D != _DYNAMIC_ )  {
            char info[512];
            sprintf( info, "<MESO> DeviceVector %s must be instantiated without a templaterized dimension to use set_d()", tag().c_str() );
            error->one( FLERR, info );
        }
    	_d = d;
    	_ptrs.grow( _d );
    	size_t n = _n;
    	grow(0);
    	grow(n);
    }
    inline size_t n() const
    {
        return _n;
    }
    inline TYPE** ptrs() const
    {
        return _ptrs;
    }
    inline size_t size() const
    {
        return _n * _d * sizeof( TYPE );
    }
    inline size_t typesize() const
    {
        return sizeof( TYPE );
    }
    inline std::string tag() const
    {
        return _tag;
    }

    DeviceVector<TYPE, D>* grow( size_t n, bool copy = true, bool zero = false )
    {
        if( n ) {
            if( !_ptr ) _ptr = meso_device->malloc_device_pitch<TYPE>( _tag, _pitch, n, _d );
            else meso_device->realloc_device_pitch( _tag, _ptr, _pitch, n, _d, copy, zero );
            _pitch /= sizeof( TYPE );
            _n = n;
            set_ptrs();
        } else {
            destroy();
        }
        return this;
    }
    void set( TYPE v, cudaStream_t stream = 0, size_t offset = 0, size_t count = SIZE_MAX )
    {
        if( count == SIZE_MAX ) count = _n - offset;
        if( offset + count > _n ) {
            char info[512];
            sprintf( info, "<MESO> %s device buffer size %lu smaller than requested %lu", tag().c_str(), _n, offset + count );
            error->one( FLERR, info );
        }
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_buffer_vset<TYPE>, 0 );
        if( stream != 0 ) {
            gpu_buffer_vset <<< grid_cfg.x, grid_cfg.y, 0, stream >>>( _ptrs.ptr(), v, offset, count, _d );
            verify( ( cudaPeekAtLastError() ) );
        } else {
            gpu_buffer_vset <<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>> ( _ptrs.ptr(), v, offset, count, _d );
            verify( ( cudaStreamSynchronize( meso_device->stream() ) ) );
        }
    }

protected:
    inline void destroy()
    {
        if( _ptr ) meso_device->free( _ptr );
        _ptr = NULL;
        _n = _pitch = 0;
    }
    inline void set_ptrs()
    {
        std::vector<TYPE*> ptrs(_d);
        for( int i = 0 ; i < _d ; i++ ) ptrs[i] = _ptr + i * _pitch;
        _ptrs.upload( ptrs.data(), _d );
    }

    uint _d;
    size_t _pitch, _n;
    std::string _tag;
    TYPE* _ptr;
    DeviceScalar<TYPE*> _ptrs;
};

class TextureObject: protected Pointers, protected MesoPointers
{
public:
    TextureObject( LAMMPS *lmp, const std::string tag ) :
        Pointers( lmp ),
        MesoPointers( lmp ),
        _tag( tag ), _ref( "" ),
        _res_ptr( NULL ),
        _res_sz( 0 ),
        _tex( 0 ) {}
    TextureObject( const TextureObject &another ) :
        Pointers( another.lmp ),
        MesoPointers( another.lmp ),
        _tag( "" ), _ref( "" ),
        _res_ptr( NULL ),
        _res_sz( 0 ),
        _tex( 0 )
    {
        *this = another;
    }
    inline const TextureObject& operator = ( const TextureObject& another )
    {
        _tag     = another._tag;
        _ref     = another._ref;
        _res_ptr = another._res_ptr;
        _res_sz  = another._res_sz;
        _tex     = another._tex;
        return *this;
    }
    ~TextureObject()
    {
        /*do not unbind on destructiuon*/
    }

    inline std::string tag() const
    {
        return _tag;
    }
    inline std::string ref() const
    {
        return _ref;
    }
    inline texobj tex() const
    {
        return _tex;
    }
    inline operator texobj() const
    {
        return tex();
    }

    template<typename TYPE>
    void bind( const DeviceScalar<TYPE> &mem )
    {
        if( _res_ptr == mem.ptr() && _res_sz == mem.size() ) return;
        else if( bound() ) unbind();

        cudaResourceDesc res_desc;
        memset( &res_desc, 0, sizeof( res_desc ) );
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = mem.ptr();
        res_desc.res.linear.desc = cudaCreateChannelDesc<TYPE>();
        res_desc.res.linear.sizeInBytes = mem.size();

        cudaTextureDesc tex_desc;
        memset( &tex_desc, 0, sizeof( tex_desc ) );
        tex_desc.readMode = cudaReadModeElementType;

        verify( ( cudaCreateTextureObject( &_tex, &res_desc, &tex_desc, NULL ) ) );
        _res_ptr = ( void* ) mem.ptr();
        _res_sz = mem.size();
    }
    inline void unbind()
    {
        if( bound() ) {
            meso_device->sync_device(); // make sure previous kernels using this object has completed
            verify( ( cudaDestroyTextureObject( _tex ) ) );
            _tex = 0;
            _ref = "";
            _res_ptr = NULL;
            _res_sz = 0;
        }
    }
    inline bool bound()
    {
        return _tex ? true : false;
    }

protected:
    std::string _tag, _ref;
    void *_res_ptr;
    size_t _res_sz;
    texobj _tex;
};

template< template<typename> class VTYPE, typename DTYPE >
class Pointer1DBase
{
public:
    Pointer1DBase() {
        _ptr = NULL;
    }
    inline VTYPE<DTYPE>* operator = ( VTYPE<DTYPE>* ptr ) {
        return _ptr = ptr;
    }
    inline operator DTYPE* () const {
        if( !_ptr ) raise( SIGSEGV );
        return _ptr->operator DTYPE * ();
    }
    inline operator VTYPE<DTYPE>& () const {
        if( !_ptr ) raise( SIGSEGV );
        return *_ptr;
    }
    inline VTYPE<DTYPE>& operator * () const {
        if( !_ptr ) raise( SIGSEGV );
        return *_ptr;
    }
protected:
    VTYPE<DTYPE>* _ptr;
};

template< template<typename, uint> class VTYPE, typename DTYPE, uint D >
class Pointer2DBase
{
public:
    Pointer2DBase() {
        _ptr = NULL;
    }
    inline VTYPE<DTYPE, D>* operator = ( VTYPE<DTYPE, D>* ptr ) {
        return _ptr = ptr;
    }
    inline DTYPE* operator []( uint i ) const {
        if( !_ptr ) raise( SIGSEGV );
        return _ptr->operator[]( i );
    }
    inline DTYPE** ptrs() const {
        if( !_ptr ) raise( SIGSEGV );
        return _ptr->ptrs();
    }
    inline operator VTYPE<DTYPE, D>& () const {
        if( !_ptr ) raise( SIGSEGV );
        return *_ptr;
    }
    inline VTYPE<DTYPE, D>& operator * () const {
        if( !_ptr ) raise( SIGSEGV );
        return *_ptr;
    }
protected:
    VTYPE<DTYPE, D>* _ptr;
};

template< typename DTYPE >
struct PtrDeviceScalar: public Pointer1DBase<DeviceScalar, DTYPE> {
	using Pointer1DBase<DeviceScalar, DTYPE>::operator =;
};

template< typename DTYPE >
struct PtrHostScalar: public Pointer1DBase<HostScalar, DTYPE> {
	using Pointer1DBase<HostScalar, DTYPE>::operator =;
};

template< typename DTYPE >
struct PtrDevicePitched: public Pointer1DBase<DevicePitched, DTYPE> {
    size_t pitch() const {
        return this->_ptr->pitch();
    }
    using Pointer1DBase<DevicePitched, DTYPE>::operator =;
};

template<typename DTYPE, uint D = _DYNAMIC_>
struct PtrDeviceVector: public Pointer2DBase<DeviceVector, DTYPE, D> {
	using Pointer2DBase<DeviceVector, DTYPE, D>::operator =;
};

template<typename TYPE>
std::ostream &operator << ( std::ostream &out, const DeviceScalar<TYPE> &s ) {
	std::vector<TYPE> v( s.n() );
	cudaDeviceSynchronize();
	s.download( v.data(), v.size() );
	cudaDeviceSynchronize();
	out<< "[";
	for(int i=0;i<v.size();i++) out << v[i] << ", ";
	out<< "]";
	return out;
}

}

#endif
