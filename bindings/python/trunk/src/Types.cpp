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

void init_typename2typeid()
{
    typename2typeid[typeid(int8_t).name()  ]  = BYTE;
    typename2typeid[typeid(uint8_t).name() ]  = UBYTE;
    typename2typeid[typeid(int16_t).name() ]  = SHORT;
    typename2typeid[typeid(uint16_t).name()]  = USHORT;
    typename2typeid[typeid(int32_t).name() ]  = INT;
    typename2typeid[typeid(uint32_t).name()]  = UINT;
    typename2typeid[typeid(int64_t).name() ]  = LONG;
    typename2typeid[typeid(uint64_t).name()]  = ULONG;
    typename2typeid[typeid(float).name()   ]  = FLOAT;
    typename2typeid[typeid(double).name()  ]  = DOUBLE;
    typename2typeid[typeid(std::string).name() ]  = STRING;
}

void init_typeid2size()
{
    typeid2size[BYTE] = sizeof(int8_t);
    typeid2size[UBYTE] =  sizeof(uint8_t);
    typeid2size[SHORT] = sizeof(int16_t);
    typeid2size[USHORT] = sizeof(uint16_t);
    typeid2size[INT]  = sizeof(int32_t);
    typeid2size[UINT]  = sizeof(uint32_t);
    typeid2size[LONG]  = sizeof(int64_t);
    typeid2size[ULONG] = sizeof(uint64_t);
    typeid2size[FLOAT] = sizeof(float);
    typeid2size[DOUBLE] = sizeof(double);
    typeid2size[STRING] = sizeof(std::string);
}

void init_typeid2numpytc()
{
    typeid2numpytc[BYTE] = NPY_BYTE;
    typeid2numpytc[UBYTE] = NPY_UBYTE;
    typeid2numpytc[SHORT] = NPY_SHORT;
    typeid2numpytc[USHORT] = NPY_USHORT;
    typeid2numpytc[INT] = NPY_INT;
    typeid2numpytc[UINT] = NPY_UINT;
    typeid2numpytc[LONG]  = NPY_LONG;
    typeid2numpytc[ULONG] = NPY_ULONG;
    typeid2numpytc[FLOAT] = NPY_FLOAT;
    typeid2numpytc[DOUBLE] = NPY_DOUBLE;
}

void init_typeid2numpystr()
{
    typeid2numpystr[BYTE] = "int8";
    typeid2numpystr[UBYTE] = "uint8";
    typeid2numpystr[SHORT] = "int16";
    typeid2numpystr[USHORT] = "uint16";
    typeid2numpystr[INT] = "int32";
    typeid2numpystr[UINT] = "uint32";
    typeid2numpystr[LONG] = "int64";
    typeid2numpystr[ULONG] = "uint64";
    typeid2numpystr[FLOAT] = "float32";
    typeid2numpystr[DOUBLE]  = "float64";
    typeid2numpystr[STRING] = "string";
}
