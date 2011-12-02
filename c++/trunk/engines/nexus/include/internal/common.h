#ifndef __CDMA_NXSCOMMON_H__
#define __CDMA_NXSCOMMON_H__

#define NXS_FACTORY_NAME "NxsFactory"
#include <nxfile.h>
#include <yat/memory/SharedPtr.h>
#include <yat/utils/String.h>

typedef yat::SharedPtr<NexusFile, yat::Mutex> NexusFilePtr;

#endif