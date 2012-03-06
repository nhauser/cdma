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
#  if defined(WIN32) && defined(CDMA_DLL)
#    if defined (CDMA_BUILD)
#      define CDMA_DECL __declspec(dllexport)
#    else
#      define CDMA_DECL __declspec(dllimport)
#    endif
#  else
#    define CDMA_DECL
#  endif
#endif

#include <cdma/TraceDebug.h>
#include <list>
#include <yat/memory/SharedPtr.h>

#define DECLARE_SHARED_PTR(x)\
  typedef yat::SharedPtr<x, yat::Mutex> x##Ptr

#define DECLARE_CLASS_SHARED_PTR(x)\
  class x;\
  DECLARE_SHARED_PTR(x)

#define DECLARE_WEAK_PTR(x)\
  typedef yat::WeakPtr<x, yat::Mutex> x##WPtr

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

/// Generic types
typedef std::list<std::string> StringList;
typedef yat::SharedPtr<StringList, yat::Mutex> StringListPtr;

#endif // __CDMA_COMMON_H__