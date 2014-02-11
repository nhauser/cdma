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
import java.util.List;

import org.cdma.dictionary.Context;
import org.cdma.dictionary.IPluginMethod;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.internal.dictionary.solvers.Solver;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.navigation.NxsGroup;

import fr.soleil.nexus.NexusNode;
import fr.soleil.nexus.PathNexus;

/**
 * Analyze the path previously entered, to extract the specified attribute
 * path is under the form: "/node1/node2@attribute_name=value"
 * 
 * Only nodes with that particular attribute are returned.
 * 
 * @note: the character '*' is a wildcard for text completion both in name and value of attribute
 * 
 * @author rodriguez
 */
public final class AttributeFilter implements IPluginMethod {

    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> result = filterOnAttribute(context);
        context.setContainers(result);

    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    /**
     * Stack all found data items to construct an aggregated NxsDataItem
     * 
     * @param context @return
     */
    public List<IContainer> filterOnAttribute(Context context) {
        LogicalGroup group = (LogicalGroup) context.getCaller();
        List<Solver> solvers = context.getSolver();

        // The last Solver was a Path
        Path path = solvers.get(solvers.size() - 2).getPath();
        String addr = path.toString();

        // Extract the subpart corresponding to attribute and value
        String[] attr = addr.substring(addr.lastIndexOf('@') + 1).split("=");
        addr = addr.substring(0, addr.lastIndexOf('@'));

        NexusNode[] nodes = PathNexus.splitStringToNode(addr);

        NxsDataset handler = ((NxsDataset) group.getDataset());

        List<NxsDataItem> list = getAllDataItems(handler, nodes);
        List<IContainer> items = new ArrayList<IContainer>();
        String expected = attr[1].replace("*", ".*");
        for (NxsDataItem item : list) {
            IAttribute attribute = item.getAttribute(attr[0]);
            if (attribute != null) {
                String value = attribute.getStringValue();
                if (value != null && value.matches(expected)) {
                    items.add(item);
                }
            }
        }
        return items;
    }

    /**
     * Recursively explore the tree represented by the given array of nodes
     */
    protected static List<NxsDataItem> getAllDataItems(NxsDataset handler, NexusNode[] nodes) {
        return getAllDataItems((NxsGroup) handler.getRootGroup(), nodes, 0);
    }

    /**
     * Recursively explore the tree represented by the given array of nodes beginning
     * the exploration at the depth node
     */
    private static List<NxsDataItem> getAllDataItems(NxsGroup entryPoint, NexusNode[] nodes, int depth) {
        List<NxsDataItem> result = new ArrayList<NxsDataItem>();

        if (depth < nodes.length) {
            NexusNode node = nodes[depth];
            if (depth < nodes.length - 1) {
                List<IGroup> groups = entryPoint.getGroupList();

                for (IGroup current : groups) {
                    NexusNode leaf = ((NxsGroup) current).getPathNexus().getCurrentNode();
                    if (leaf.matchesPartNode(node)) {
                        result.addAll(getAllDataItems((NxsGroup) current, nodes, depth + 1));
                    }
                }
            } else {
                List<IDataItem> items = entryPoint.getDataItemList();
                for (IDataItem current : items) {
                    NexusNode leaf = ((NxsDataItem) current).getNexusItems()[0].getPath().getCurrentNode();
                    if (leaf.matchesPartNode(node)) {
                        result.add((NxsDataItem) current);
                    }
                }
            }
        }
        return result;
    }

}
