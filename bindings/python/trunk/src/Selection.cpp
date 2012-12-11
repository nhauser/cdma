/*
 * (c) Copyright 2012 DESY, Eugen Wintersberger <eugen.wintersberger@desy.de>
 *
 * This file is part of cdma-python.
 *
 * cdma-python is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * cdma-python is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************
 *
 * Created on: Jul 03, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */
#include "Selection.hpp"


//-----------------------------------------------------------------------------
size_t size(const Selection &sel)
{
    size_t s=0;
    for(auto iter = sel.shape().begin();
             iter != sel.shape().end();++iter)
        s += *iter;

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
    for(auto iter = s.offset().begin();
             iter != s.offset().end(); ++iter)
        o<<*iter<<" ";

    o<<std::endl<<"stride: [ ";
    for(auto iter = s.offset().begin();
             iter != s.offset().end(); ++iter)
        o<<*iter<<" ";

    o<<std::endl<<"shape: [ ";
    for(auto iter = s.shape().begin();
             iter != s.shape().end(); ++iter)
        o<<*iter<<" ";

    return o;
}

        
