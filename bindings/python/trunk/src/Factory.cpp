/*Factory wrapper*/


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

