#ifndef __CDMA_TYPEDETECTOR_H__
#define __CDMA_TYPEDETECTOR_H__

#include <typeinfo>
#include <nxfile.h>

namespace cdma
{

//==============================================================================
/// TypeDetector
/// ....
//==============================================================================
class TypeDetector
{
public:
  static const std::type_info& detectType( NexusDataType type );
  static const NexusDataType&  detectType( const std::type_info& type);
  template<typename T> static const NexusDataType& detectType( T* type );
  template<typename T> static T* allocate(NexusDataType type, unsigned int length);

};

template<typename T> const NexusDataType& TypeDetector::detectType( T* type )
{
  return detectType( typeid(*type) );
}


template<typename T> T* TypeDetector::allocate(NexusDataType type, unsigned int length)
{
  switch( type )
  {
    case NX_INT16:
      return new short[length];
      break;
    case NX_UINT16:
      return new unsigned short[length];
      break;
    case NX_UINT32:
      return new unsigned long[length];
      break;
    case NX_INT32:
      return new long[length];
      break;
    case NX_FLOAT32:
      return new float[length];
      break;
    case NX_INT64:
      return new yat::int64[length];
      break;
    case NX_UINT64:
      return new yat::uint64[length];
      break;
    case NX_FLOAT64:
      return new double[length];
      break;
    default:  // CHAR, NX_INT8, NX_UINT8
      return new char[length];
  }
}

} // namespace

#endif // __CDMA_TYPEDETECTOR_H__