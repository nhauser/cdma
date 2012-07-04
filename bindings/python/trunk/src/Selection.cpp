#include "Selection.hpp"


//-----------------------------------------------------------------------------
size_t size(const Selection &sel)
{
    size_t s=0;
    for(auto v: sel.shape()) s+=v;
    return s;
}

//-----------------------------------------------------------------------------
size_t span(const Selection &sel)
{
    size_t s=0;
    for(size_t i=0;i<sel.rank();i++)
        s += (sel.shape()[i]*sel.stride()[i]);

    return s;
}

//-----------------------------------------------------------------------------
void set_selection_parameters_from_index(size_t i,const extract<size_t> &index,
                                         std::vector<size_t> &offset,
                                         std::vector<size_t> &stride,
                                         std::vector<size_t> &shape)
{
    offset[i] = index; 
    shape[i]  = 1; 
    stride[i] = 1;
}

//-----------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &o,const Selection &s)
{
    o<<"Selection of rank: "<<s.rank()<<std::endl;
    o<<"offset: [ ";
    for(auto &v: s.offset()) o<<v<<" ";
    o<<std::endl<<"stride: [ ";
    for(auto &v: s.stride()) o<<v<<" ";
    o<<std::endl<<"shape: [ ";
    for(auto &v: s.shape()) o<<v<<" ";
    return o;
}

        
