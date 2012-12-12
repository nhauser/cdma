#include "Types.hpp"


//-----------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &o,const TypeID &tid)
{
    return o<<TypeUtility::typeid2numpystr(tid);
}

//-----------------------------------------------------------------------------
bool operator<(TypeID a,TypeID b) { return int(a)<int(b); }

//-----------------------------------------------------------------------------
bool operator>(TypeID a,TypeID b) { return int(a)>int(b); }

//-----------------------------------------------------------------------------
bool operator<=(TypeID a,TypeID b) { return int(a)<=int(b); }

//-----------------------------------------------------------------------------
bool operator>=(TypeID a,TypeID b) { return int(a)>=int(b); }

//-----------------------------------------------------------------------------
TypeID TypeUtility::typename2typeid(const std::string &tname)
{
    if(tname == typeid(int8_t).name()) return BYTE;
    if(tname == typeid(uint8_t).name()) return UBYTE;
    if(tname == typeid(int16_t).name()) return SHORT;
    if(tname == typeid(uint16_t).name()) return USHORT;
    if(tname == typeid(int32_t).name()) return INT;
    if(tname == typeid(uint32_t).name()) return UINT;
    if(tname == typeid(int64_t).name()) return LONG;
    if(tname == typeid(uint64_t).name()) return ULONG;
    if(tname == typeid(float).name()) return FLOAT;
    if(tname == typeid(double).name()) return DOUBLE;
    if(tname == typeid(std::string).name()) return STRING;
}

//-----------------------------------------------------------------------------
size_t TypeUtility::typeid2size(const TypeID &tid)
{
    switch(tid)
    {
        case(BYTE):  return sizeof(int8_t);
        case(UBYTE): return sizeof(uint8_t);
        case(SHORT): return sizeof(int16_t);
        case(USHORT): return sizeof(uint16_t);
        case(INT): return sizeof(int32_t);
        case(UINT): return sizeof(uint32_t);
        case(LONG): return sizeof(int64_t);
        case(ULONG): return sizeof(uint64_t);
        case(FLOAT): return sizeof(float);
        case(DOUBLE): return sizeof(double);
        case(STRING): return sizeof(std::string);
    }
}

//-----------------------------------------------------------------------------
int TypeUtility::typeid2numpytc(const TypeID &tid)
{
    switch(tid)
    {
        case(BYTE): return NPY_BYTE;
        case(UBYTE): return NPY_UBYTE;
        case(SHORT): return NPY_SHORT;
        case(USHORT): return NPY_USHORT;
        case(INT): return NPY_INT;
        case(UINT): return NPY_UINT;
        case(LONG): return NPY_LONG;
        case(ULONG): return NPY_ULONG;
        case(FLOAT): return NPY_FLOAT;
        case(DOUBLE): return NPY_DOUBLE;
    } 
}

//-----------------------------------------------------------------------------
std::string TypeUtility::typeid2numpystr(const TypeID &tid)
{
    switch(tid)
    {
        case(BYTE): return "int8";
        case(UBYTE): return "uint8";
        case(SHORT): return "int16";
        case(USHORT): return "uint16";
        case(INT): return "int32";
        case(UINT): return "uint32";
        case(LONG): return "int64";
        case(ULONG): return "uint64";
        case(FLOAT): return "float32";
        case(DOUBLE): return  "float64";
        case(STRING): return "string";
    }
}


