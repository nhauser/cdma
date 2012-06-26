/*wrapper for IDataset objects*/
#ifndef __DATASETWRAPPER_HPP__
#define __DATASETWRAPPER_HPP__


#include<iostream>
#include<list>
#include<boost/python.hpp>

using namespace boost::python;

#include<cdma/Factory.h>
#include<cdma/IFactory.h>
#include<cdma/navigation/IDataset.h>

using namespace cdma;

#include "GroupWrapper.hpp"

class DatasetWrapper
{
    private:
        IDatasetPtr _dataset; //! pointer to the dataset
        IFactoryPtr _factory; //! pointer to the factory
        GroupWrapper _root_group; //! pointer to the root group of the dataset
    public:
        //===============constructors and destructor===========================
        //! default constructor
        DatasetWrapper():
            _dataset(nullptr),
            _factory(nullptr),
            _root_group()
        {}

        //---------------------------------------------------------------------
        //! standard constructor
        DatasetWrapper(IDatasetPtr dptr,IFactoryPtr fptr):
            _dataset(dptr),
            _factory(fptr),
            _root_group(GroupWrapper(_dataset->getRootGroup()))
        { }

        //---------------------------------------------------------------------
        //! destructor
        ~DatasetWrapper() {} 

        //==============assignment operators===================================
        DatasetWrapper &operator=(const DatasetWrapper &ds)
        {
            if(this == &ds) return *this;
            _dataset = ds._dataset;
            _factory = ds._factory;
            _root_group = ds._root_group;

            return *this;
        }

        //=================implement dataset interface=========================
        //! get dataset title
        std::string getTitle() const { return _dataset->getTitle(); }

        //---------------------------------------------------------------------
        //! get dataset location
        std::string getLocation() const { return _dataset->getLocation(); }

        //---------------------------------------------------------------------
        //! get list of childs
        tuple childs() const { return _root_group.childs(); }

        //---------------------------------------------------------------------
        //! get an object from the dataset
        object __getitem__(const std::string &path) const
        {
            return _root_group.__getitem__(path);
        }


};

#endif
