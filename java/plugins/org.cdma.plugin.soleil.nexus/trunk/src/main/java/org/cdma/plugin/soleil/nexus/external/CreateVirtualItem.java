/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus.external;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.array.NxsArray;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataItem;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;
import org.cdma.plugin.soleil.nexus.utils.NxsNode;
import org.cdma.plugin.soleil.nexus.utils.NxsPath;
import org.cdma.utils.Utilities.ModelType;

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
    public void execute(final Context context) throws CDMAException {
        List<IContainer> inList = context.getContainers();
        List<IContainer> outList = new ArrayList<IContainer>();

        NxsDataItem item;
        NxsPath path;
        NxsArray array;
        String name;
        for (IContainer container : inList) {
            if (container.getModelType().equals(ModelType.Group)) {
                NxsGroup group = (NxsGroup)container;
                
                name = container.getName();
                path = new NxsPath(group.getNxsPath());
                path.addNode(new NxsNode(container.getShortName()));
                item = new NxsDataItem(name, (NxsDataset) container.getDataset());
                array = new NxsArray(name.toCharArray(), new int[] { name.length() });
                item.setShortName(container.getShortName());
                item.setDataset(container.getDataset());
                item.setNxsPath(path);
                item.setParent((IGroup) container);
                item.setCachedData(array, false);
                for (IAttribute attr : container.getAttributeList()) {
                    item.addOneAttribute(attr);
                }
                outList.add(item);
            } else {
                outList.add(container);
            }
        }

        // Update context
        context.setContainers(outList);
    }

}
