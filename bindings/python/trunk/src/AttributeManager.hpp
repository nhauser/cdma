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

#include <list>
#include<cdma/navigation/IContainer.h>
#include "AttributeWrapper.hpp"
#include "TupleIterator.hpp"

using namespace cdma;

/*! 
\ingroup utility_classes
\brief manages attributes of an object

This type provides an interface to the attribute attached to an object
satisfying the IContainer interface. 
Each type implementing the IContainer interface has a public member variable
of type AttributeManager. The type implements basic sequence interface for
Python. Thus, one can iterate over such an attribute in python with
\code
item = ....
for attr in item.attrs:
    print attr
\endcode
An individual attribute can be picked using the [] operator
\code
print item.attrs["temperatur"]
\endcode
If the attribute does not exist a KeyError exception is thrown. 
*/
template<typename CPTR> class AttributeManager
{
    private:
        CPTR _ptr; //!< pointer to the container object
    public:
        //===================public types======================================
        typedef AttributeWrapper value_type; //!< value type of the container
        //===================public constructors===============================
        //! default constructor
        AttributeManager():_ptr(NULL) {}

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
        /*! 
        \brief retrieve attribute by name

        Return an attribute by name. If the attribute does not exist a KeyError
        is thrown in Python. 
        \param name the attributes name
        \return requested attribute
        */
        AttributeWrapper __getitem__str(const std::string &name) const 
        {
            IAttributePtr ptr = NULL;
            ptr = this->_ptr->getAttribute(name);
            if(!ptr)
                throw_PyKeyError("Attribute ["+name+"] not found!");

            return AttributeWrapper(this->_ptr->getAttribute(name));
        }

        //---------------------------------------------------------------------
        /*!
        \brief return attribute by index

        Return an attribute by its index. This method will not be exposed to
        Python. It is used to satisfy the interface required by the PyIterator
        template. 
        \return instance of AttributeWrapper
        */
        AttributeWrapper operator[](size_t i) const
        {
            size_t cnt=0;

            std::list<IAttributePtr> alist = this->_ptr->getAttributeList();

            for(auto iter = alist.begin(); iter != alist.end();++iter)
            {
                if(cnt == i) return AttributeWrapper(*iter);
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

        //----------------------------------------------------------------------
        /*!
        \brief return iterator

        Returns an interator over the attributes managed by this instance of
        AttributeManager. 
        \return iterator
        */
        TupleIterator create_iterator() const
        {
            list l;
            std::list<IAttributePtr> alist = this->_ptr->getAttributeList();
            for(auto iter = alist.begin(); iter != alist.end();++iter)
            {
                l.append(AttributeWrapper(*iter));
            }

            return TupleIterator(tuple(l),0);
        }


};

//===================implementation of template methods========================
/*! 
\ingroup wrapper_classes
\brief wrapper function for an attribute manager

Creates a wrapper for an AttributeManager.
\param name the name under which the class will appear in Python
*/
template<typename CPTR> void wrap_attribute_manager(const char *name)
{
    //create the attribute manager type
    class_<AttributeManager<CPTR>>(name)
        .def("__getitem__",&AttributeManager<CPTR>::__getitem__str)
        .def("__len__",&AttributeManager<CPTR>::size)
        .def("__iter__",&AttributeManager<CPTR>::create_iterator)
        ;
}

#endif
