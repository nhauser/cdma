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

#ifndef __GROUPWRAPPER_HPP__
#define __GROUPWRAPPER_HPP__

#include<boost/python.hpp>
using namespace boost::python;

#include<cdma/navigation/IGroup.h>
using namespace cdma;

#include "Container.hpp"
#include "AttributeManager.hpp"
#include "TupleIterator.hpp"

/*! 
\brief wrapper for IGroupPtr

Wraps IGroupPtr. 
*/
class GroupWrapper:public ContainerWrapper<IGroupPtr>
{
    public:
        //==================constructors and destructor========================
        //! default constructor
        GroupWrapper():ContainerWrapper<IGroupPtr>() {}

        //---------------------------------------------------------------------
        //! standard constructor
        GroupWrapper(IGroupPtr g):ContainerWrapper<IGroupPtr>(g) {}

        //---------------------------------------------------------------------
        //! destructor
        ~GroupWrapper() {}

        //==================assignment operators===============================
        //! copy assignment operator
        GroupWrapper &operator=(const GroupWrapper &g)
        {
            if(this == &g) return *this;
            ContainerWrapper<IGroupPtr>::operator=(g);
            attrs = g.attrs;
            return *this;
        }

        //===================data access methods===============================
        /*! 
        \brief return child object
        
        Returns a child object of the group determined by name. If the object
        cannot be found the Python KeyError exception is thrown.
        \param name the child objects name
        \return instance of the child object as Python object
        */
        object __getitem__(const std::string &name) const;

        //---------------------------------------------------------------------
        //! returns a tuple of all childs of a group
        tuple childs() const;


        //---------------------------------------------------------------------
        //! return the parent group 
        GroupWrapper parent() const;

        //---------------------------------------------------------------------
        //! return the root group
        GroupWrapper root() const;

        //---------------------------------------------------------------------
        //! create iterator
        TupleIterator create_iterator() ;


        //---------------------------------------------------------------------
        //! string representation for Python
        std::string __str__() const;
};
        

#endif
