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

#ifndef __DATASETWRAPPER_HPP__
#define __DATASETWRAPPER_HPP__


#include<iostream>
#include<list>
#include<boost/python.hpp>

using namespace boost::python;

#include<cdma/factory/Factory.h>
#include<cdma/navigation/IDataset.h>

using namespace cdma;

#include "GroupWrapper.hpp"

/*! 
\ingroup wrapper_classes
\brief dataset wrapper

Wraps a IDatasetPtr pointer and the factory pointer belonging to it. The dataset
object is considered as an entry level object. Thus only a view methods are
accessible to the Python world.
*/
class DatasetWrapper
{
    private:
        IDatasetPtr _dataset; //! pointer to the dataset
        GroupWrapper _root_group; //! pointer to the root group of the dataset
    public:
        //===============constructors and destructor===========================
        //! default constructor
        DatasetWrapper():
            _dataset(NULL),
            _root_group()
        {}

        //---------------------------------------------------------------------
        //! standard constructor
        DatasetWrapper(IDatasetPtr dptr):
            _dataset(dptr),
            _root_group(GroupWrapper(_dataset->getRootGroup()))
        { }

        //---------------------------------------------------------------------
        //! destructor
        ~DatasetWrapper() {} 

        //==============assignment operators===================================
        //! copy assignment operator
        DatasetWrapper &operator=(const DatasetWrapper &ds)
        {
            if(this == &ds) return *this;
            _dataset = ds._dataset;
            _root_group = ds._root_group;

            return *this;
        }

        //=================implement dataset interface=========================
        //! get dataset title
        std::string getTitle() const 
        { 
            return _dataset->getTitle(); 
        }

        //---------------------------------------------------------------------
        //! get dataset location
        std::string getLocation() const 
        { 
            return _dataset->getLocation(); 
        }

        //---------------------------------------------------------------------
        //! get root group
        GroupWrapper getRoot() const 
        { 
            return GroupWrapper(_dataset->getRootGroup()); 
        }
};

#endif
