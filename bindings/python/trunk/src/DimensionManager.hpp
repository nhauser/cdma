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
 * Created on: Jul 04, 2011
 *     Author: Eugen Wintersberger <eugen.wintersberger@desy.de>
 */

#ifndef __DIMENSIONMANAGER_HPP__
#define __DIMENSIONMANAGER_HPP__

#include<cdma/navigation/IDimension.h>
#include<cdma/navigation/IDataItem.h>
#include "DimensionWrapper.hpp"
#include "Exceptions.hpp"

using namespace cdma;

/*! 
\ingroup utility_classes
\brief DimensionManager class

Every DataItemWrapper has a public instance of DimensionManager attached to it.
This object basically behaves like a list. The length of the list is the number
of dimensions of the data item. Each list entry is a list of DimensionWrapper
objects which can be associated with the very dimension the list entry belongs
to.
*/
class DimensionManager
{
    private:
        IDataItemPtr _ptr; //!< pointer to the data item the manger belongs to
    public:
        //================constructors and destructor==========================
        //! default constructor
        DimensionManager():_ptr(NULL) {}

        //---------------------------------------------------------------------
        //! copy constructor
        DimensionManager(const DimensionManager &d):_ptr(d._ptr) {}

        //---------------------------------------------------------------------
        //! standard constructor
        DimensionManager(IDataItemPtr ptr):_ptr(ptr) {}

        //--------------------------------------------------------------------
        //! destructor
        ~DimensionManager() {}

        //===================assignment operators=============================
        //! copy assignment operator
        DimensionManager &operator=(const DimensionManager &d)
        {
            if(this != &d) return *this;
            _ptr = d._ptr;
            return *this;
        }

        //=====================================================================
        /*! 
        \brief number of entries

        Returns the size of the list which is in fact the rank of the data item. 
        \return number of dimensions
        */
        size_t size() const
        {
            size_t cnt =0;
            while(_ptr->getDimensions(cnt).size()!=0) cnt++;

            return cnt;
        }

        //---------------------------------------------------------------------
        /*! 
        \brief return dimensions for axis i

        Return a tuple of dimensions associated with axis i of the data item 
        object.
        \param i index of axis
        \return tuple of dimensions 
        */
        tuple dimensions(size_t i) const
        {
            list l;
            if(i>=size())
                throw_PyIndexError("Axis index out of bounds!");

            std::list<IDimensionPtr> dlist = _ptr->getDimensions(i);
            for(std::list<IDimensionPtr>::iterator iter = dlist.begin(); 
                iter != dlist.end(); ++iter)
            {
                l.append(DimensionWrapper(*iter));
            }
            return tuple(l);
        }

        
};

//wrapper function template
void wrap_dimensionmanager();

#endif
