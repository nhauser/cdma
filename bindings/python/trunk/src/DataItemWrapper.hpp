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

#ifndef __DATAITEMWRAPPER_HPP__
#define __DATAITEMWRAPPER_HPP__

#include<cdma/navigation/IDataItem.h>

#include "Container.hpp"
#include "ArrayWrapper.hpp"
#include "WrapperHelpers.hpp"
#include "DimensionManager.hpp"

/*! 
\brief wraps IDataItemPtr

Wrapper type for IDataItemPtr.
*/
class DataItemWrapper:public ContainerWrapper<IDataItemPtr>
{
    public:
        //=======================public members================================
        DimensionManager dim; //!< dimension manager for this data item
        //================constructors and destructor==========================
        //! default constructor
        DataItemWrapper():ContainerWrapper<IDataItemPtr>(),dim() {}

        //---------------------------------------------------------------------
        //! standard constructor
        DataItemWrapper(IDataItemPtr item):
            ContainerWrapper<IDataItemPtr>(item),
            dim(item)
        {}

        //---------------------------------------------------------------------
        //! destructor
        ~DataItemWrapper() {}

        //====================assignment operator==============================
        //! copy assignment operator
        DataItemWrapper &operator=(const DataItemWrapper &item)
        {
            if(this == &item) return *this;
            ContainerWrapper<IDataItemPtr>::operator=(item);
            dim = item.dim;
            return *this;
        }
        
        //======================inquery methods================================
        /*! 
        \brief get number of dimensions

        Return the number of dimensions from a DataItem object.
        \return number of dimensions
        */
        size_t rank() const { return ptr()->getRank(); }

        //---------------------------------------------------------------------
        /*! 
        \brief return data item shape

        Return the shape of a data item as a Python list. Each entry describes
        the number of elements along this particular dimension.
        \return Python list with number of elements
        */
        std::vector<size_t> shape() const;

        //---------------------------------------------------------------------
        /*! 
        \brief return number of elements

        Return the numbner of elements stored in the data item.
        \return number of elements
        */
        size_t size() const { return ptr()->getSize(); }

        //---------------------------------------------------------------------
        /*! 
        \brief return unit

        Return the unit of the data stored in this item as string.
        \return unit 
        */
        std::string unit() const { return ptr()->getUnitsString(); }

        //---------------------------------------------------------------------
        /*!
        \brief get description

        Return the description of the data item. 
        \return description string
        */
        std::string description() const { return ptr()->getDescription(); }

        //---------------------------------------------------------------------
        /*!
        \brief return type id

        Return the ID of the data type used to store data.
        \return type id
        */
        TypeID type() const;

        //---------------------------------------------------------------------
        /*!
        \brief returns a list of dimensions

        Returns a list of all dimensions associated with this data item. 
        \return list of dimensions
        */
        std::list<IDimensionPtr> dimensions() const;

        //---------------------------------------------------------------------
        /*! 
        \brief read scalar data

        Reads data for a scalar data item. The return type is determined by the
        template parameter.
        \return object of type T
        */
        template<typename T> T get() const
        {
            return ptr()->getData()->getValue<T>();
        }

        //---------------------------------------------------------------------
        /*!
        \brief read array data

        Reads data in cases where the data item is non-scalar. The offset
        determines the index offset from where to start reading and shape gives
        the number of elements along each dimension.
        \param offset index offset
        \param shape number of elements along each dimension
        \return instance of ArrayWrapper with data
        */
        ArrayWrapper get(const std::vector<size_t> &offset,
                         const std::vector<size_t> & shape);

        
        
};




#endif
