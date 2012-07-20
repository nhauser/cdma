#ifndef __CDMA_TypeUtils_H__
#define __CDMA_TypeUtils_H__

#include <typeinfo>
#include <nxfile.h>
#include <cdma/exception/Exception.h>
#include <internal/common.h>

namespace cdma
{
namespace nexus
{

//==============================================================================
/// TypeUtils
/// Internal class allowing type manipulation and data conversion
//==============================================================================
class CDMA_NEXUS_DECL TypeUtils
{
public:
  static const std::type_info& toCType( NexusDataType type );
  static const std::type_info& toRawCType( NexusDataType type );
  static int sizeOf(NexusDataType type);
  static NexusDataType toNexusDataType( const std::type_info& type );
  template<typename T> static NexusDataType toNeXusDataType();
  template<typename T> static T valueToType( void* value_ptr, const std::type_info& Ctype );
};

//----------------------------------------------------------------------------
// TypeUtils::valueToType
//----------------------------------------------------------------------------
template<typename T> T TypeUtils::valueToType( void* value_ptr, const std::type_info& Ctype )
{
  if( typeid(T) == Ctype )
    return *( (T*)( value_ptr ) );

  else if( Ctype == typeid(short) )
    return T( *(short*)( value_ptr ) );

  else if( Ctype == typeid(unsigned short) )
    return T( *(unsigned short*)( value_ptr ) );

  else if( Ctype == typeid(long) )
    return T( *(long*)( value_ptr ) );

  else if( Ctype == typeid(unsigned long) )
    return T( *(unsigned long*)( value_ptr ) );

  else if( Ctype == typeid(float) )
    return T( *(float*)( value_ptr ) );

  else if( Ctype == typeid(double) )
    return T( *(double*)( value_ptr ) );

  else if( Ctype == typeid(int) )
    return T( *(int*)( value_ptr ) );

  else if( Ctype == typeid(unsigned int) )
    return T( *(unsigned int*)( value_ptr ) );

  else
    throw cdma::Exception("INVALID_TYPE", "Cannot convert data to the requested type", 
                          "Array::getValue");
}

template <>
inline NexusDataType TypeUtils::toNeXusDataType<int>()
{
  return NX_INT32;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<unsigned int>()
{
  return NX_UINT32;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<char>()
{
  return NX_INT8;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<unsigned char>()
{
  return NX_UINT8;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<short>()
{
  return NX_INT16;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<unsigned short>()
{
  return NX_UINT16;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<long>()
{
  if( sizeof(long) == 4 )
  {
    return NX_INT32;
  }
  else
  {
    return NX_INT64;
  }
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<unsigned long>()
{
  if( sizeof(unsigned long) == 4 )
  {
    return NX_UINT32;
  }
  else
  {
    return NX_UINT64;
  }
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<float>()
{
  return NX_FLOAT32;
}
template <>
inline NexusDataType TypeUtils::toNeXusDataType<double>()
{
  return NX_FLOAT64;
}
template <typename T>
inline NexusDataType TypeUtils::toNeXusDataType()
{
  throw cdma::Exception("UNRECOGNIZED TYPE", "Unable to map <TYPE> to Nexus type", "TypeUtils::toNeXusDataType");
}

} // namespace nexus
} // namespace cdma

#endif // __CDMA_TypeUtils_H__
