//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.array.NxsArray;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.utils.Utilities.ModelType;

import fr.soleil.nexus.PathNexus;

/**
 * Create a list of IDataItem that are empty from a IGroup list. Created items have no data linked.
 * 
 * @param context
 * @throws CDMAException
 */

public class CreateVirtualItem implements IPluginMethod {

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        NxsDataItem item;
        PathNexus path;
        NxsArray array;
        String name;
        for (IContainer container : inList) {
            if (container.getModelType().equals(ModelType.Group)) {
            	name = container.getName();
            	path = new PathNexus(PathNexus.splitStringToNode(container.getLocation()));
                item = new NxsDataItem();
                array = new NxsArray(name.toCharArray(), new int[] {name.length()});
                item.setName(name);
                item.setShortName(container.getShortName());
                item.setDataset(container.getDataset());
                item.getNexusItems()[0].setPath(path);
                item.setParent( (IGroup) container);
                item.setCachedData(array, false);
                for (IAttribute attr : container.getAttributeList()) {
                    item.addOneAttribute(attr);
                }
                outList.add(item);
            }
            else {
                outList.add(container);
            }
        }

        // Update context
        context.setContainers(outList);
    }

}
