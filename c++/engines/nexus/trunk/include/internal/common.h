#ifndef __CDMA_NXS_COMMON_H__
#define __CDMA_NXS_COMMON_H__

#define NXS_FACTORY_NAME "NxsFactory"
#include <nxfile.h>
#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>
#include <yat/utils/String.h>

// CDMA Core
#include <cdma/Common.h>

#ifndef CDMA_NEXUS_DECL
#  if defined(WIN32) && defined(CDMA_NEXUS_DLL)
#    if defined (CDMA_NEXUS_BUILD)
#      define CDMA_NEXUS_DECL __declspec(dllexport)
#    else
#      define CDMA_NEXUS_DECL __declspec(dllimport)
#    endif
#  else
#    define CDMA_NEXUS_DECL
#  endif
#endif


namespace cdma
{
namespace nexus
{
DECLARE_CLASS_SHARED_WEAK_PTR(Group);
DECLARE_CLASS_SHARED_WEAK_PTR(Dataset);
DECLARE_SHARED_PTR(NexusFile);
}
}

#endif