#include "Types.hpp"


//-----------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &o,const TypeID &tid)
{
    return o<<typeid2numpystr[tid];
}

//-----------------------------------------------------------------------------
bool operator<(TypeID a,TypeID b) { return int(a)<int(b); }

//-----------------------------------------------------------------------------
bool operator>(TypeID a,TypeID b) { return int(a)>int(b); }

//-----------------------------------------------------------------------------
bool operator<=(TypeID a,TypeID b) { return int(a)<=int(b); }

//-----------------------------------------------------------------------------
bool operator>=(TypeID a,TypeID b) { return int(a)>=int(b); }

