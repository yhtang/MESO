#ifdef _USE_NVTX

#ifndef LMP_MESO_NVTX
#define LMP_MESO_NVTX

#include <stdint.h>
#include "nvToolsExt.h"

inline void nvtx_push_range( const char name[], uint32_t color ) {
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;
	nvtxRangePushEx(&eventAttrib);
}

inline void nvtx_pop_range( const char name[] ) {
	nvtxRangePop();
}

inline uint32_t nvtx_color( int i, double alpha = 1.0 ) {
	const static int nmax = 15;
	// static uint32_t colors[] = { 0x00E61919, 0x00E66419, 0x00E6AE19, 0x00D3E619, 0x0088E619, 0x003EE619, 0x0019E63F, 0x0019E68A, 0x0019E6D4, 0x0019ADE6, 0x001962E6, 0x001B19E6, 0x006519E6, 0x00B019E6, 0x00E619D1 };
	static uint32_t colors[] = {
			0x00E61919,
			0x00E6C919,
			0x0054E619,
			0x0019E68E,
			0x00198EE6,
			0x005419E6,
			0x00E619C9,
			0x00E61919,
			0x00E6C919,
			0x0054E619,
			0x0019E68E,
			0x00198EE6,
			0x005419E6,
			0x00E619C9,
			0x00E61919,
	};
	return colors[ i % nmax ];
}

#endif

#else

inline void nvtx_push_range( const char name[], uint32_t color ) {}
inline void nvtx_pop_range( const char name[] ) {}
inline uint32_t nvtx_color( int i, double alpha = 1.0 ) { return 0xFFFFFFFF; }

#endif
