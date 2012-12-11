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

#include<iostream>
#include<list>

#include<boost/python.hpp>


using namespace boost::python;

#include<cdma/factory/Factory.h>
#include<cdma/navigation/IDataset.h>

using namespace cdma;

#include "DatasetWrapper.hpp"

/*! 
\brief CDMA factory class

Wraps the Factory singleton. This class must be used to initialize and cleanup
the entire CDMA stack. This class will never be used by the user but is for
internal use of the module only.
*/
class FactoryWrapper
{
    public:
        //---------------------------------------------------------------------
        /*!
        \brief initialize CDMA library

        Initialization method for the entire CDMA library. 
        \param path path to plugins
        */
        static void init(const std::string &path) { Factory::init(path); } 

        //---------------------------------------------------------------------
        /*! 
        \brief cleanup library

        Cleanup the library - this must be called before the module gets
        unloaded (in other words when the interpreter exits).
        */
        static void cleanup() { Factory::cleanup(); }

        //---------------------------------------------------------------------
        /*! 
        \brief open a new dataset

        Opens a new dataset. The plugin to be used is determined automatically. 
        \param path the path to the dataset to open
        \return dataset instance
        */
        static DatasetWrapper open_dataset(const std::string &path) 
        {
            IDatasetPtr p(Factory::openDataset(path)); 
            return DatasetWrapper(p);
        }
};

//=============================================================================

/*! 
\brief create factory Python type
*/
void wrap_factory()
{
    class_<FactoryWrapper,boost::noncopyable>("_factory",no_init)
        .def("init",&FactoryWrapper::init)
        .def("cleanup",&FactoryWrapper::cleanup)
        .def("open_dataset",&FactoryWrapper::open_dataset)
        .staticmethod("cleanup")
        .staticmethod("init")
        .staticmethod("open_dataset")
        ;
    
}

