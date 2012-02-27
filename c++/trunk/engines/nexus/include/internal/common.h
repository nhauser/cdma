#ifndef __CDMA_NXSCOMMON_H__
#define __CDMA_NXSCOMMON_H__

#define NXS_FACTORY_NAME "NxsFactory"
#include <nxfile.h>
#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>
#include <yat/utils/String.h>

// CDMA Core
#include <cdma/Common.h>

namespace cdma
{
DECLARE_CLASS_SHARED_WEAK_PTR(NxsGroup);
DECLARE_CLASS_SHARED_WEAK_PTR(NxsDataset);
DECLARE_SHARED_PTR(NexusFile);
}

#endif