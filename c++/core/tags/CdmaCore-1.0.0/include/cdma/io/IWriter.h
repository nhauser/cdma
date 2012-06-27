// USELESS FOR NOW
// To restore replace "//*" by "/*" and "* //" by "*/"
/*
//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IWRITER_H__
#define __CDMA_IWRITER_H__

#include <string>

#include "exception/Exception.h"

namespace CDMACore
{
	interface IWriter {
		public:
			//Virtual destructor
			virtual ~IWriter() {};

	//**
	 * Open the storage, for example, the file handler. Any output action will
	 * require the storage to be open.
	 *
	 * @throw  Exception
	 *             failed to open the storage
	 * //
	virtual void open() throw ( Exception ) = 0;

	//**
	 * Check if the storage is open for output.
	 *
	 * @return true or false
	 * //
	virtual bool isOpen() = 0;

	//**
	 * Add a group to the root of the storage. The root has an X-path of '/'.
	 * This has the same performance as
	 * {@link #writeToRoot(IGroup group, bool force)}, where force is set to
	 * be false.
	 *
	 * @param group
	 *            Gumtree group object
	 * @throw  Exception
	 *             failed to write the group
	 * //
	virtual void writeToRoot(IGroup& group) throw ( Exception ) = 0;

	//**
	 * Write a group to the root of the storage. The root has an X-path of '/'.
	 * When a group with a same name already exists under the root node, write
	 * the contents of the GDM group under the target group node. For conflict
	 * data item, check the force switch. If it is set to be true, overwrite the
	 * contents under the group. Otherwise raise an exception. <br>
	 * See {@link #writeGroup(String, IGroup, bool)} for more information.
	 *
	 * @param group
	 *            GDM group object
	 * @param force
	 *            if allow overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void writeToRoot(IGroup& group, bool force) throw ( Exception ) = 0;

	//**
	 * Write a data item to the root of the storage. If a data node with the
	 * same name already exists, raise an exception.
	 *
	 * @param dataItem
	 *            GDM data item
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void writeToRoot(IDataItem& dataItem) throw ( Exception ) = 0;

	//**
	 * Write a data item to the root of the storage. If force is true, overwrite
	 * the conflicting node.
	 *
	 * @param dataItem
	 *            GDM data item
	 * @param force
	 *            if allow overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeToRoot(IDataItem dataItem, bool force)
			throw ( Exception );

	//**
	 * Write an attribute to the root of the storage. If an attribute node
	 * already exists in the root of the storage, raise an exception.
	 *
	 * @param attribute
	 *            GDM attribute
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void writeToRoot(IAttribute& attribute) throw ( Exception ) = 0;

	//**
	 * Write an attribute to the root of the storage. If an attribute node
	 * already exists, check the force switch. If it is true, overwrite the
	 * node. Otherwise raise an exception.
	 *
	 * @param attribute
	 *            GDM attribute
	 * @param force
	 *            if allow overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeToRoot(IAttribute attribute, bool force)
			throw ( Exception );

	//**
	 * Write a group under the node with a given X-path. If a group node with
	 * the same name already exists, this will not overwrite any existing
	 * contents under the node. When conflicting happens, raise an exception.
	 *
	 * @param parentPath
	 *            x-path as a string object
	 * @param group
	 *            GDM group
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void writeGroup(string parentPath, IGroup& group) throw ( Exception ) = 0;

	//**
	 * Write a group under the node of a given X-path. If a group node with the
	 * same name already exists, check the 'force' switch. If it is true,
	 * overwrite any conflicting contents under the node. Otherwise raise an
	 * exception for conflicting.
	 *
	 * @param parentPath
	 *            x-path as a string object
	 * @param group
	 *            GDM group
	 * @param force
	 *            if allow overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeGroup(string parentPath, IGroup group, bool force)
			throw ( Exception );

	//**
	 * Write a data item under a group node with a given X-path. If a data item
	 * node already exists there, raise an exception.
	 *
	 * @param parentPath
	 *            x-path as a string object
	 * @param dataItem
	 *            GDM data item
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeDataItem(string parentPath, IDataItem dataItem)
			throw ( Exception );

	//**
	 * Write a data item under a group node with a given X-path. If a data item
	 * node already exists there, check the 'force' switch. If it is true,
	 * overwrite the node. Otherwise raise an exception.
	 *
	 * @param parentPath
	 *            string value
	 * @param dataItem
	 *            IDataItem object
	 * @param force
	 *            true or false
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeDataItem(string parentPath, IDataItem dataItem, bool force)
			throw ( Exception );

	//**
	 * Write an attribute under a node with a given X-path. The parent node can
	 * be either a group node or a data item node. If an attribute node with the
	 * same name already exists, raise an exception.
	 *
	 * @param parentPath
	 *            x-path as a string object
	 * @param attribute
	 *            GDM attribute
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeAttribute(string parentPath, IAttribute attribute)
			throw ( Exception );

	//**
	 * Write an attribute to the node with a given X-path. The node can be
	 * either a group node or a data item node. If an attribute with an existing
	 * name already exists, raise an exception.
	 *
	 * @param parentPath
	 *            x-path as a string object
	 * @param attribute
	 *            GDM attribute
	 * @param force
	 *            if allowing overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeAttribute(string parentPath, IAttribute attribute, bool force)
			throw ( Exception );

	//**
	 * Write an empty group under a group node with a given X-path. If a group
	 * node with the same name already exists, check the 'force' switch. If it
	 * is true, remove all the contents of the group node.
	 *
	 * @param xPath
	 *            as a string object
	 * @param groupName
	 *            short name as a string object
	 * @param force
	 *            if allow overwriting
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	void writeEmptyGroup(string xPath, string groupName, bool force)
			throw ( Exception );

	//**
	 * Remove a group with a given X-path from the storage.
	 *
	 * @param groupPath
	 *            as a string object
	 * //
	virtual void removeGroup(string groupPath) = 0;

	//**
	 * Remove a data item with a given X-path from the storage.
	 *
	 * @param dataItemPath
	 *            as a string object
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void removeDataItem(string dataItemPath) throw ( Exception ) = 0;

	//**
	 * Remove an attribute with a given X-path from the storage.
	 *
	 * @param attributePath
	 *            x-path as a string object
	 * @throw  Exception
	 *             failed to write to hdf file
	 * //
	virtual void removeAttribute(string attributePath) throw ( Exception ) = 0;

	//**
	 * Check if a group exists in a given X-path.
	 *
	 * @param xPath
	 *            as a string object
	 * @return true or false
	 * //
	virtual bool isGroupExist(string xPath) = 0;

	//**
	 * Check if a group exists under certain group node with a given X-path.
	 *
	 * @param parentPath
	 *            the X-path of the parent group
	 * @param groupName
	 *            the name of the target group
	 * @return true or false
	 * //
	virtual bool isGroupExist(string parentPath, string groupName) = 0;

	//**
	 * Check if a data item exists with a given X-path.
	 *
	 * @param xPath
	 *            as a string object
	 * @return true or false
	 * //
	virtual bool isDataItemExist(string xPath) = 0;

	//**
	 * Check if a data item exists under a parent group with a given X-path.
	 *
	 * @param parentPath
	 *            x-path of the parent group as a string object
	 * @param dataItemName
	 *            name of the target data item
	 * @return true or false
	 * //
	virtual bool isDataItemExist(string parentPath, string dataItemName) = 0;

	//**
	 * Check if an attribute exist with a given xpath.
	 *
	 * @param xPath
	 *            x-path as a string object
	 * @return true or false
	 * //
	virtual bool isAttributeExist(string xPath) = 0;

	//**
	 * Check if the attribute with a given name already exists.
	 *
	 * @param parentPath
	 *            string object
	 * @param attributeName
	 *            string object
	 * @return true or false
	 * //
	virtual bool isAttributeExist(string parentPath, string attributeName) = 0;

	//**
	 * Close the file handler. Unlock the file.
	 * //
	virtual void close() = 0;
	};
} //namespace CDMACore
#endif //__CDMA_IWRITER_H__
*/
