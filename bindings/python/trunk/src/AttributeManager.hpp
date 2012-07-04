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
 * Created on: Jun 26, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __ATTRIBUTE_MANAGER_HPP__
#define __ATTRIBUTE_MANAGER_HPP__

#include<cdma/navigation/IContainer.h>
#include "AttributeWrapper.hpp"

using namespace cdma;

template<typename CPTR> class AttributeManager
{
    private:
        CPTR _ptr; //!< pointer to the container object
    public:
        //===================public constructors===============================
        //! default constructor
        AttributeManager():_ptr(nullptr) {}

        //---------------------------------------------------------------------
        //! copy constructor
        AttributeManager(const AttributeManager<CPTR> &m):_ptr(m._ptr) 
        { } 

        //---------------------------------------------------------------------
        //! default constructor
        AttributeManager(CPTR ptr):_ptr(ptr) {}

        //---------------------------------------------------------------------
        //! destructor
        ~AttributeManager() {}

        //====================assignment operators=============================
        //! copy assignment operator
        AttributeManager<CPTR> &operator=(const AttributeManager<CPTR> &m)
        {
            if(this == &m) return *this;
            _ptr = m._ptr;
            return *this;
        }

        //====================attribute related methods========================
        AttributeWrapper __getitem__str(const std::string &name) const 
        {
            IAttributePtr ptr = nullptr;
            ptr = this->_ptr->getAttribute(name);
            if(!ptr)
                throw_PyKeyError("Attribute ["+name+"] not found!");

            return AttributeWrapper(this->_ptr->getAttribute(name));
        }

        //---------------------------------------------------------------------
        size_t __len__() const
        {
            return this->_ptr->getAttributeList().size();
        }

        
};

//===================implementation of template methods========================
template<typename CPTR> void wrap_attribute_manager(const char *name)
{
    class_<AttributeManager<CPTR>>(name)
        .def("__getitem__",&AttributeManager<CPTR>::__getitem__str)
        .def("__len__",&AttributeManager<CPTR>::__len__)
        ;
}

#endif
