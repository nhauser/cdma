#include "GroupWrapper.hpp"
#include "DataItemWrapper.hpp"

#include <cdma/navigation/IContainer.h>

//================implementation of class methods==============================
object GroupWrapper::__getitem__(const std::string &path) const
{
    //check groups
    for(auto v: ptr()->getGroupList())
        if(path == v->getShortName()) return object(new GroupWrapper(v));

    //check data items
    for(auto v: ptr()->getDataItemList())
        if(path == v->getShortName()) return object(new DataItemWrapper(v));


    //throw an exception here
}

//----------------------------------------------------------------------------
GroupWrapper GroupWrapper::parent() const
{
    return GroupWrapper(ptr()->getParent());
}

//----------------------------------------------------------------------------
GroupWrapper GroupWrapper::root() const
{
    return GroupWrapper(ptr()->getRoot());
}

//----------------------------------------------------------------------------
tuple GroupWrapper::childs() const
{
    list l;
    
    for(auto v: ptr()->getGroupList()) l.append(GroupWrapper(v));
    for(auto v: ptr()->getDataItemList()) l.append(DataItemWrapper(v));

    return tuple(l);
}

//----------------------------------------------------------------------------
tuple GroupWrapper::groups() const
{
    list l;
    for(auto v:ptr()->getGroupList()) l.append(GroupWrapper(v));
    return tuple(l);
}

//----------------------------------------------------------------------------
tuple GroupWrapper::items() const
{
    list l;
    for(auto v:ptr()->getDataItemList()) l.append(DataItemWrapper(v));
    return tuple(l);
}

//====================helper function to create python class==================
void wrap_group()
{

    wrap_container<IGroupPtr>("GroupContainer");
    wrap_attribute_manager<IGroupPtr>("GroupAttributeManager");

    class_<GroupWrapper,bases<ContainerWrapper<IGroupPtr>> >("Group")
        .def_readwrite("attrs",&GroupWrapper::attrs)
        .add_property("parent",&GroupWrapper::parent)
        .add_property("root",&GroupWrapper::root)
        .add_property("childs",&GroupWrapper::childs)
        .add_property("items",&GroupWrapper::items)
        .add_property("gruops",&GroupWrapper::groups)
        .def("__getitem__",&GroupWrapper::__getitem__)
        ;
}


