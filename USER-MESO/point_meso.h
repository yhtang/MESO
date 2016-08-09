/*
 * point.h
 *
 *  vector class using the expression template technique
 *  Created and authored by Yu-Hang Tang on from 2014-11 to 2015-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present source code for any purpose including, but not limited to,
 *  scientific publications and production software before getting a written permission
 *  from the author of this file.
 */

#ifndef POINT_H_
#define POINT_H_

#include<array>
#include<cmath>
#include<algorithm>
#include<cassert>
#include<iostream>

namespace ermine {

using uint = unsigned int;
using ulong = unsigned long;

template<typename T1, typename T2> struct same_type { static bool const yes = false; };
template<typename T> struct same_type<T,T> { static bool const yes = true; };

// Reference:
// Expression template: http://en.wikipedia.org/wiki/Expression_templates
// CRTP: http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

/*---------------------------------------------------------------------------
                              Interface
---------------------------------------------------------------------------*/

template<class VEX, typename SCALAR, uint D>
struct vec_exp {
	using TYPE_ = SCALAR;
	static const uint D_ = D;

	// the only work in constructor is type checking
	inline vec_exp() {
		static_assert( same_type<TYPE_, typename VEX::TYPE_>::yes, "Vector element type mismatch" );
		static_assert( D_ == VEX::D_, "Vector dimensionality mismatch" );
	}

	// dereferencing using static polymorphism
	inline SCALAR operator[] (uint i) const {
		assert( i < D );
		return static_cast<VEX const &>(*this)[i];
	}

	inline operator VEX      & ()       { return static_cast<VEX      &>(*this); }
	inline operator VEX const& () const { return static_cast<VEX const&>(*this); }

	inline uint d() const { return D_; }
};

/*---------------------------------------------------------------------------
                                 Container
---------------------------------------------------------------------------*/

template<typename SCALAR, uint D=3U>
struct vector : public vec_exp<vector<SCALAR,D>, SCALAR, D> {
protected:
	SCALAR x[D];
public:
	using TYPE_ = SCALAR;
	static const uint D_ = D;

	// default constructor
	inline vector() {}
	// construct from scalar constant
	inline vector(SCALAR const s) { for(uint i = 0 ; i < D ; i++) x[i] = s; }
	inline vector(int    const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline vector(uint   const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline vector(long   const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	inline vector(ulong  const s) { for(uint i = 0 ; i < D ; i++) x[i] = SCALAR(s); }
	// construct from C-array
	inline vector(SCALAR const *ps) { for(uint i = 0 ; i < D ; i++) x[i] = *ps++; }
	// construct from parameter pack
	// 'head' differentiate it from constructing from vector expression
	template<typename ...T> inline vector(SCALAR const head, T const ... tail ) {
		std::array<TYPE_,D_> s( { head, static_cast<SCALAR>(tail)... } );
		for(uint i = 0 ; i < D ; i++) x[i] = s[i];
	}
	// construct from any vector expression
	template<class E> inline vector( const vec_exp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] = u[i];
	}

	// Vector must be assignable, while other expressions may not
	inline SCALAR      & operator [] (uint i)       { assert( i < D ); return x[i]; }
	inline SCALAR const& operator [] (uint i) const { assert( i < D ); return x[i]; }

	// STL-style direct data accessor
	inline SCALAR      * data()       { return x; }
	inline SCALAR const* data() const { return x; }

	// assign from any vector expression
	template<class E> inline vector & operator += ( const vec_exp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] += u[i];
		return *this;
	}
	template<class E> inline vector & operator -= ( const vec_exp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] -= u[i];
		return *this;
	}
	template<class E> inline vector & operator *= ( const vec_exp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] *= u[i];
		return *this;
	}
	template<class E> inline vector & operator /= ( const vec_exp<E,SCALAR,D> &u ) {
		for(uint i = 0 ; i < D ; i++) x[i] /= u[i];
		return *this;
	}
	// conventional vector-scalar operators
	inline vector & operator += ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] += u;
		return *this;
	}
	inline vector & operator -= ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] -= u;
		return *this;
	}
	inline vector & operator *= ( SCALAR const u ) {
		for(uint i = 0 ; i < D ; i++) x[i] *= u;
		return *this;
	}
	inline vector & operator /= ( SCALAR const u ) {
		return operator *= ( SCALAR(1)/u );
	}

	// special vectors
	static inline vector const & zero() {
		static vector zero_(0);
		return zero_;
	}
};

/*---------------------------------------------------------------------------
                         Arithmetic Functors
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR, uint D>
struct VecAdd: public vec_exp<VecAdd<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecAdd( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] + v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecSub: public vec_exp<VecSub<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecSub( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] - v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct VecNeg: public vec_exp<VecNeg<E,SCALAR,D>, SCALAR, D> {
	inline VecNeg( vec_exp<E,SCALAR,D> const& u ) : u_(u) {}
	inline SCALAR operator [] (uint i) const { return -u_[i]; }
protected:
	E const& u_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecMul: public vec_exp<VecMul<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecMul( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] * v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct VecScale: public vec_exp<VecScale<E,SCALAR,D>, SCALAR, D> {
	inline VecScale( vec_exp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] * a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E, typename SCALAR, uint D>
struct VecAddScalar: public vec_exp<VecAddScalar<E,SCALAR,D>, SCALAR, D> {
	inline VecAddScalar( vec_exp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] + a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E, typename SCALAR, uint D>
struct VecSubScalar: public vec_exp<VecSubScalar<E,SCALAR,D>, SCALAR, D> {
	inline VecSubScalar( vec_exp<E,SCALAR,D> const& u, SCALAR const a ) : u_(u), a_(a) {}
	inline SCALAR operator [] (uint i) const { return u_[i] - a_; }
protected:
	E      const& u_;
	SCALAR const  a_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecDiv: public vec_exp<VecDiv<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecDiv( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[i] / v_[i]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, typename SCALAR, uint D>
struct vexpr_rcp: public vec_exp<vexpr_rcp<E,SCALAR,D>, SCALAR, D> {
	inline vexpr_rcp( vec_exp<E,SCALAR,D> const& u ) : u_(u) {}
	inline SCALAR operator [] (uint i) const { return SCALAR(1)/u_[i]; }
protected:
	E const& u_;
};

template<class E, typename SCALAR, uint D>
struct VecScaleRcp: public vec_exp<VecScaleRcp<E,SCALAR,D>, SCALAR, D> {
	inline VecScaleRcp( SCALAR const a, vec_exp<E,SCALAR,D> const& u ) : a_(a), u_(u) {}
	inline SCALAR operator [] (uint i) const { return a_ / u_[i]; }
protected:
	SCALAR const  a_;
	E      const& u_;
};

template<class E1, class E2, typename SCALAR>
struct VecCross: public vec_exp<VecCross<E1,E2,SCALAR>, SCALAR, 3U> {
	inline VecCross( vec_exp<E1,SCALAR,3U> const& u, vec_exp<E2,SCALAR,3U> const& v ) : u_(u), v_(v) {}
	inline SCALAR operator [] (uint i) const { return u_[(i+1U)%3U] * v_[(i+2U)%3U] - u_[(i+2U)%3U] * v_[(i+1U)%3U]; }
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E1, class E2, typename SCALAR, uint D>
struct VecCrossAs3D: public vec_exp<VecCrossAs3D<E1,E2,SCALAR,D>, SCALAR, D> {
	inline VecCrossAs3D( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v ) : u_(u), v_(v) {
		static_assert( D > 3, "cannot do cross product for vector dimensionality < 3" );
	}
	inline SCALAR operator [] (uint i) const {
		return i < 3U ? u_[(i+1U)%3U] * v_[(i+2U)%3U] - u_[(i+2U)%3U] * v_[(i+1U)%3U] : SCALAR(0);
	}
protected:
	E1 const& u_;
	E2 const& v_;
};

template<class E, class OP, typename SCALAR, uint D>
struct VecApply1: public vec_exp<VecApply1<E,OP,SCALAR,D>, SCALAR, D> {
	inline VecApply1( vec_exp<E,SCALAR,D> const& u, OP const op ) : u_(u), o_(op) {}
	inline SCALAR operator [] (uint i) const { return o_( u_[i] ); }
protected:
	E  const& u_;
	OP const  o_;
};

template<class E1, class E2, class OP, typename SCALAR, uint D>
struct VecApply2: public vec_exp<VecApply2<E1,E2,OP,SCALAR,D>, SCALAR, D> {
	inline VecApply2( vec_exp<E1,SCALAR,D> const& u, vec_exp<E2,SCALAR,D> const& v, OP const op ) : u_(u), v_(v), o_(op) {}
	inline SCALAR operator [] (uint i) const { return o_( u_[i], v_[i] ); }
protected:
	E1 const& u_;
	E2 const& v_;
	OP const  o_;
};

/*---------------------------------------------------------------------------
                         Operator Overloads
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR, uint D> inline
VecAdd<E1, E2, SCALAR, D> operator + ( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecAdd<E1, E2, SCALAR, D>( u, v );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecSub<E1, E2, SCALAR, D> operator - ( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecSub<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecNeg<E, SCALAR, D> operator - ( vec_exp<E,SCALAR,D> const &u ) {
	return VecNeg<E, SCALAR, D>( u );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecMul<E1, E2, SCALAR, D> operator * ( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecMul<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator * ( vec_exp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecScale<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator * ( SCALAR const a, vec_exp<E,SCALAR,D> const &u ) {
	return VecScale<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> operator / ( vec_exp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecScale<E, SCALAR, D>( u, SCALAR(1)/a );
}

template<class E, typename SCALAR, uint D> inline
VecAddScalar<E, SCALAR, D> operator + ( vec_exp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecAddScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecAddScalar<E, SCALAR, D> operator + ( SCALAR const a, vec_exp<E,SCALAR,D> const &u ) {
	return VecAddScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecSubScalar<E, SCALAR, D> operator - ( vec_exp<E,SCALAR,D> const &u, SCALAR const a ) {
	return VecSubScalar<E, SCALAR, D>( u, a );
}

template<class E, typename SCALAR, uint D> inline
VecSubScalar<E, SCALAR, D> operator - ( SCALAR const a, vec_exp<E,SCALAR,D> const &u ) {
	return VecAddScalar<E, SCALAR, D>( -u, a );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecDiv<E1, E2, SCALAR, D> operator / ( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecDiv<E1, E2, SCALAR, D>( u, v );
}

template<class E, typename SCALAR, uint D> inline
VecScaleRcp<E, SCALAR, D> operator / ( SCALAR const a, vec_exp<E,SCALAR,D> const &u ) {
	return VecScaleRcp<E, SCALAR, D>( a, u );
}

/*---------------------------------------------------------------------------
                         Math functions
---------------------------------------------------------------------------*/

template<class E1, class E2, typename SCALAR> inline
VecCross<E1, E2, SCALAR> cross( vec_exp<E1,SCALAR,3U> const &u, vec_exp<E2,SCALAR,3U> const &v ) {
	return VecCross<E1, E2, SCALAR>( u, v );
}

template<class E1, class E2, typename SCALAR, uint D> inline
VecCrossAs3D<E1, E2, SCALAR, D> cross( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecCrossAs3D<E1, E2, SCALAR, D>( u, v );
}

// generic reduction template
template<class E, class OP, typename SCALAR, uint D> inline
SCALAR reduce( vec_exp<E,SCALAR,D> const &u, OP const & op ) {
	SCALAR core( u[0] );
	for(uint i = 1 ; i < D ; i++) core = op( core, u[i] );
	return core;
}

// biggest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR max( vec_exp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a>b?a:b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR min( vec_exp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a<b?a:b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR sum( vec_exp<E,SCALAR,D> const &u ) {
	return reduce( u, [](SCALAR a, SCALAR b){return a+b;} );
}

// smallest element within a vector
template<class E, typename SCALAR, uint D> inline
SCALAR mean( vec_exp<E,SCALAR,D> const &u ) {
	return sum(u) / SCALAR(D);
}

// inner product
template<class E1, class E2, typename SCALAR, uint D> inline
SCALAR dot( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return sum( u * v );
}

// square of L2 norm
template<class E, typename SCALAR, uint D> inline
SCALAR normsq( vec_exp<E,SCALAR,D> const &u ) {
	return sum( u * u );
}

// L2 norm
template<class E, typename SCALAR, uint D> inline
SCALAR norm( vec_exp<E,SCALAR,D> const &u ) {
	return std::sqrt( normsq(u) );
}

template<class E, typename SCALAR, uint D> inline
VecScale<E, SCALAR, D> normalize( vec_exp<E,SCALAR,D> const &u ) {
	return VecScale<E, SCALAR, D>( u, SCALAR(1)/norm(u) );
}

// element-wise arbitrary function applied for each element
template<class E, class OP, typename SCALAR, uint D> inline
VecApply1<E, OP, SCALAR, D> apply( vec_exp<E,SCALAR,D> const &u, OP const& op ) {
	return VecApply1<E, OP, SCALAR, D>( u, op );
}

// element-wise arbitrary function applied element-wisely between 2 vectors
template<class E1, class E2, class OP, typename SCALAR, uint D> inline
VecApply2<E1, E2, OP, SCALAR, D> apply( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v, OP const& op ) {
	return VecApply2<E1, E2, OP, SCALAR, D>( u, v, op );
}

// element-wise flooring down
template<class E, typename SCALAR, uint D> inline
VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D> floor( vec_exp<E,SCALAR,D> const &u ) {
	return VecApply1<E, SCALAR(*)(SCALAR), SCALAR, D>( u, [](SCALAR s){return std::floor(s);} );
}

// element-wise pick bigger
template<class E1, class E2, typename SCALAR, uint D> inline
VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D> max( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D>( u, v, [](SCALAR s,SCALAR t){ return s>t?s:t;} );
}

// element-wise pick smaller
template<class E1, class E2, typename SCALAR, uint D> inline
VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D> min( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &v ) {
	return VecApply2<E1, E2, SCALAR(*)(SCALAR,SCALAR), SCALAR, D>( u, v, [](SCALAR s,SCALAR t){ return s<t?s:t;} );
}

template<class E1, class E2, class E3, typename SCALAR, uint D> inline
auto clamp( vec_exp<E1,SCALAR,D> const &u, vec_exp<E2,SCALAR,D> const &l, vec_exp<E3,SCALAR,D> const &r )
-> decltype( min(max(u,l),r) )
{
	return min(max(u,l),r);
}

/*---------------------------------------------------------------------------
                         I/O functions
---------------------------------------------------------------------------*/

template<class E, typename SCALAR, uint D> inline
std::ostream& operator <<( std::ostream &out, vec_exp<E,SCALAR,D> const &u ) {
	for(int i = 0 ; i < D ; i++) out<<u[i]<<' ';
	return out;
}

template<typename SCALAR, uint D> inline
std::istream& operator >>( std::istream &in, vector<SCALAR,D> &u ) {
	for(int i = 0 ; i < D ; i++) in>>u[i];
	return in;
}


}

#endif /* POINT_H_ */
