//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_COMMON_H__
#define __CDMA_COMMON_H__

# ifndef CDMA_DECL
#   if defined(WIN32) && defined(CDMA_DLL)
#     define CDMA_DECL_EXPORT __declspec(dllexport)
#     define CDMA_DECL_IMPORT __declspec(dllimport)
#     if defined (CDMA_BUILD)
#       define CDMA_DECL CDMA_DECL_EXPORT
#     else
#       define CDMA_DECL CDMA_DECL_IMPORT
#     endif
#   else
#     define CDMA_DECL
#     define CDMA_DECL_EXPORT
#     define CDMA_DECL_IMPORT
#   endif
# endif

#include <cdma/TraceDebug.h>
#include <list>

/// @cond internal

// Plateform & capabilities detection
#if defined (_WIN64) || defined (WIN64) || defined (_WIN32) || defined (WIN32)
  #define CDMA_WINDOWS
  #if (_MSC_VER >= 1600)
    #define CDMA_STD_SMART_PTR
  #endif
  #if (_MSC_VER >= 1700)
    #define CDMA_CPP_11
  #endif
#elif (defined _linux_ || defined __linux__)
  #define CDMA_LINUX
  #if ((__STDC_VERSION__ >= 201112) || (__cplusplus >= 201103 ))
    #define CDMA_CPP_11
    #define CDMA_STD_SMART_PTR
  #endif
#endif

// Smart pointer helpers
#ifdef CDMA_STD_SMART_PTR
  #include <memory>
  #define DECLARE_SHARED_PTR(x)\
    typedef std::shared_ptr<x> x##Ptr
  #define DECLARE_WEAK_PTR(x)\
    typedef std::weak_ptr<x> x##WPtr
#else
  #include <yat/memory/SharedPtr.h>
  #define DECLARE_SHARED_PTR(x)\
    typedef yat::SharedPtr<x, yat::Mutex> x##Ptr
  #define DECLARE_WEAK_PTR(x)\
    typedef yat::WeakPtr<x, yat::Mutex> x##WPtr
#endif


#define DECLARE_CLASS_SHARED_PTR(x)\
  class x;\
  DECLARE_SHARED_PTR(x)

#define DECLARE_CLASS_WEAK_PTR(x)\
  class x;\
  DECLARE_WEAK_PTR(x)

#define DECLARE_SHARED_WEAK_PTR(x)\
  DECLARE_SHARED_PTR(x);\
  DECLARE_WEAK_PTR(x)

#define DECLARE_CLASS_SHARED_WEAK_PTR(x)\
  class x;\
  DECLARE_SHARED_PTR(x);\
  DECLARE_WEAK_PTR(x)

/// @endcond internal

// Generic types
/// List of std::string objects
typedef std::list<std::string> StringList;
/// Declaration of shared pointer on StringList
DECLARE_SHARED_PTR(StringList);

#endif // __CDMA_COMMON_H__
