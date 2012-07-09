#include "Types.hpp"


//-----------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &o,const TypeID &tid)
{
    return o<<typeid2numpystr[tid];
}
