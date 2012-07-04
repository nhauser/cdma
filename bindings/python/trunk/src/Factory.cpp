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

#include<cdma/Factory.h>
#include<cdma/IFactory.h>
#include<cdma/navigation/IDataset.h>

using namespace cdma;

#include "DatasetWrapper.hpp"

class FactoryWrapper
{
    public:
        static void init(const std::string &path) { Factory::init(path); } 
        static void cleanup() { Factory::cleanup(); }
        static DatasetWrapper open_dataset(const std::string &path) 
        {
            std::cout<<"opening dataset ..."<<std::endl;
            std::pair<IDatasetPtr,IFactoryPtr> p =
                Factory::openDataset(yat::URI(path)); 
            std::cout<<"look at data ..."<<std::endl;
            IDatasetPtr dataset = p.first;
            return DatasetWrapper(p.first,p.second);
        }
};

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

