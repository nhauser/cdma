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
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.exception.InvalidRangeException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.INode;
import org.cdma.internal.dictionary.solvers.Solver;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataItem;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;
import org.cdma.plugin.soleil.nexus.utils.NxsPath;

/**
 * Stack all found data items to construct an aggregated IDataItem
 * then returns the first slice of that stack
 * 
 * @param context @return
 */
public final class DataItemSlice1D implements IPluginMethod {

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> result = new ArrayList<IContainer>();
        IDataItem item = getFirstSlice(context);
        if (item != null) {
            result.add(item);
        }
        context.setContainers(result);
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    public IDataItem getFirstSlice(Context context) {
        LogicalGroup group = (LogicalGroup) context.getCaller();
        List<Solver> solvers = context.getSolver();

        // The last Solver was a Path
        Path path = solvers.get(solvers.size() - 2).getPath();
        String addr = path.toString();

        // Extract the subpart corresponding to attribute and value
        String[] attr = addr.substring(addr.lastIndexOf('@') + 1).split("=");
        addr = addr.substring(0, addr.lastIndexOf('@'));

        INode[] nodes = NxsPath.splitStringToNode(addr);

        IDataItem item = null;
        List<NxsDataItem> list = getAllDataItems((NxsDataset) group.getDataset(), nodes);
        String expected = attr[1].replace("*", ".*");
        for (NxsDataItem curItem : list) {
            String value = curItem.getAttribute(attr[0]).getStringValue();
            if (value.matches(expected)) {
                item = curItem;
                try {
                    while (item.getRank() > 1) {
                        item = item.getSlice(0, 0);
                    }
                } catch (InvalidRangeException e) {
                }
                break;
            }
        }
        return item;
    }

    /**
     * Recursively explore the tree represented by the given array of nodes
     */
    protected static List<NxsDataItem> getAllDataItems(NxsDataset handler, INode[] nodes) {
        return getAllDataItems((NxsGroup) handler.getRootGroup(), nodes, 0);
    }

    /**
     * Recursively explore the tree represented by the given array of nodes beginning
     * the exploration at the depth node
     */
    private static List<NxsDataItem> getAllDataItems(NxsGroup entryPoint, INode[] nodes, int depth) {
        List<NxsDataItem> result = new ArrayList<NxsDataItem>();

        if (depth < nodes.length) {
            INode node = nodes[depth];
            if (depth < nodes.length - 1) {
                List<IGroup> groups = entryPoint.getGroupList();

                for (IGroup current : groups) {
                    INode leaf = ((NxsGroup) current).getNxsPath().getCurrentNode();
                    if (leaf.matchesPartNode(node)) {
                        result.addAll(getAllDataItems((NxsGroup) current, nodes, depth + 1));
                    }
                }
            } else {
                List<IDataItem> items = entryPoint.getDataItemList();
                for (IDataItem current : items) {
                    INode leaf = ((NxsDataItem) current).getNxsPath().getCurrentNode();
                    if (leaf.matchesPartNode(node)) {
                        result.add((NxsDataItem) current);
                    }
                }
            }
        }
        return result;
    }

}
