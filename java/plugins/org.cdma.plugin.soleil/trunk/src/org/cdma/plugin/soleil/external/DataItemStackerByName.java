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
package org.cdma.plugin.soleil.external;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.utils.Utilities.ModelType;

/**
 * Stack all found data items that have the same short name
 * to construct a list of aggregated NxsDataItem
 */
public final class DataItemStackerByName implements IPluginMethod {

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> result = new ArrayList<IContainer>();
        result.addAll(stackDataItems(context));
        context.setContainers(result);
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    public List<IDataItem> stackDataItems(Context context) {
        IDataItem item = null;
        Map<String, List<IDataItem>> allItems = new HashMap<String, List<IDataItem>>();
        NxsDataset dataset = (NxsDataset) context.getDataset();

        // Get all previously found nodes
        List<IDataItem> result = new ArrayList<IDataItem>();
        List<IContainer> nodes = context.getContainers();

        // Fill the map of items
        for (IContainer node : nodes) {
            String shortName = node.getShortName();
            // If IDataItem
            if (node.getModelType() == ModelType.DataItem) {
                // Check it is in map
                if (!allItems.containsKey(shortName)) {
                    allItems.put(shortName, new ArrayList<IDataItem>());
                }
                // Put the item in the list
                allItems.get(shortName).add((IDataItem) node);
            }
        }

        // Construct an aggregated NxsDataItem for each list of IDataItem
        if (!allItems.isEmpty()) {
            for (Entry<String, List<IDataItem>> entry : allItems.entrySet()) {
                item = new NxsDataItem(entry.getValue().toArray(new NxsDataItem[entry.getValue().size()]),
                        dataset.getRootGroup(), dataset);
                result.add(item);
            }
        }

        return result;
    }
}
