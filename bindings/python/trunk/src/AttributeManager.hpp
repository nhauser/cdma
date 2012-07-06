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
#include "PythonIterator.hpp"

using namespace cdma;

/*! 
\brief manages attributes of an object

This type provides an interface to the attribute attached to an object
satisfying the IContainer interface. 
Each type implementing the IContainer interface has a public member variable
of type AttributeManager. 
\code

\encode
*/
template<typename CPTR> class AttributeManager
{
    private:
        CPTR _ptr; //!< pointer to the container object
    public:
        //===================public types======================================
        typedef AttributeWrapper value_type;
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
        //! retrieve attribute by name
        AttributeWrapper __getitem__str(const std::string &name) const 
        {
            IAttributePtr ptr = nullptr;
            ptr = this->_ptr->getAttribute(name);
            if(!ptr)
                throw_PyKeyError("Attribute ["+name+"] not found!");

            return AttributeWrapper(this->_ptr->getAttribute(name));
        }

        //---------------------------------------------------------------------
        //! create a list
        AttributeWrapper operator[](size_t i) const
        {
            size_t cnt=0;
            for(auto v: this->_ptr->getAttributeList())
            {
                if(cnt == i) return AttributeWrapper(v);
                cnt++;
            }

        }

        //---------------------------------------------------------------------
        /*!
        \brief number of attributes

        Returns the number of attributes managed by this instance of
        AttributeManager. 
        \return number of attributes
        */
        size_t size() const
        {
            return this->_ptr->getAttributeList().size();
        }

        PyIterator<AttributeManager> create_iterator() const
        {
            return PyIterator<AttributeManager>(*this,0);
        }


};

//===================implementation of template methods========================
template<typename CPTR> void wrap_attribute_manager(const char *name)
{
    wrap_pyiterator<AttributeManager<CPTR> >(std::string(name)+"Iterator");

    class_<AttributeManager<CPTR>>(name)
        .def("__getitem__",&AttributeManager<CPTR>::__getitem__str)
        .def("__len__",&AttributeManager<CPTR>::size)
        .def("__iter__",&AttributeManager<CPTR>::create_iterator)
        ;
}

#endif
