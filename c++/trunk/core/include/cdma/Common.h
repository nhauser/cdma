//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
// Contributors :
// See AUTHORS file 
//******************************************************************************
#ifndef __CDMA_COMMON_H__
#define __CDMA_COMMON_H__

#ifndef CDMA_DECL
#	if defined(WIN32) && defined(CDMA_DECL_DLL)
#		if defined (CDMA_DECL_BUILD)
#			define CDMA_DECL __declspec(dllexport)
#		else
#			define CDMA_DECL __declspec(dllimport)
#		endif
#	else
#		define CDMA_DECL
#	endif
#endif

#include <cdma/TraceDebug.h>

#endif // __CDMA_COMMON_H__