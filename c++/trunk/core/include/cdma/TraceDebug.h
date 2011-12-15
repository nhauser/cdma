//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
// Contributors :
// See AUTHORS file 
//******************************************************************************
#ifndef __CDMA_TRACE_DEBUG_H__
#define __CDMA_TRACE_DEBUG_H__

#include <string>
#include <iostream>
#include <iomanip>
#include <yat/threading/Utilities.h>

namespace cdma
{

#ifdef CDMA_DEBUG

  // !!! DO NOT CALL THIS FUNCTION !!!
  inline void _OutputDebug(const void* this_addr, const char* s1, const char* s2="")
  {
    std::cout << "[" 
              << std::hex
              << std::setfill('0')
              << std::setw(8)
              << yat::ThreadingUtilities::self() 
              << "][" 
              << std::hex
              << std::setfill(' ')
              << std::setw(10)
              << this_addr
              << std::dec
              << "] - "
              << s1
              << s2
              << std::endl;    
  }

  // !!! DO NOT INSTANTIATE THIS CLASS !!!
  class CDMA_DECL _FunctionTrace
  {
  private:
    const char* m_func_name;
    const void* m_this_addr;
    
  public:
    _FunctionTrace(const char *func_name, const void* this_addr) : m_func_name(func_name), m_this_addr(this_addr)
    {
      _OutputDebug(m_this_addr, "Entering ", m_func_name);
    }
    ~_FunctionTrace()
    {
      _OutputDebug(m_this_addr, "Leaving  ", m_func_name);
    }
  };
  
  /// A function trace helper
  #define CDMA_FUNCTION_TRACE(function_name) cdma::_FunctionTrace _func_trace(function_name, this)

  /// A simple trace helper
  #define CDMA_TRACE(trace) _OutputDebug(this, trace)
    
#else  
  #define CDMA_FUNCTION_TRACE(s)
  #define CDMA_TRACE(s)
#endif

} // namespace

#endif // __CDMA_TRACE_DEBUG_H__
