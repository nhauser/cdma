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
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.INode;
import org.cdma.internal.dictionary.solvers.Solver;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataItem;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.navigation.NxsGroup;
import org.cdma.plugin.soleil.utils.NxsPath;

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
     * @param context @return
     */
    public List<IContainer> filterOnAttribute(Context context) {
        LogicalGroup group = (LogicalGroup) context.getCaller();
        List<Solver> solvers = context.getSolver();

        // The last Solver was a Path
        Path path = solvers.get( solvers.size() - 2 ).getPath();
        String addr = path.toString();

        // Extract the subpart corresponding to attribute and value
        String[] attr = addr.substring(addr.lastIndexOf('@') + 1 ).split("=");
        addr = addr.substring(0, addr.lastIndexOf('@'));

        INode[] nodes = NxsPath.splitStringToNode(addr);

        NxsDataset handler = ((NxsDataset) group.getDataset());

        List<NxsDataItem> list = getAllDataItems(handler, nodes);
        List<IContainer> items = new ArrayList<IContainer>();
        String expected = attr[1].replace("*", ".*");
        for( NxsDataItem item : list ) {
            IAttribute attribute = item.getAttribute( attr[0] );
            if( attribute != null  ) {
                String value = attribute.getStringValue();
                if( value != null && value.matches(expected) ) {
                    items.add( item );
                }
            }
        }
        return items;
    }

    /**
     * Recursively explore the tree represented by the given array of nodes
     */
    protected static List<NxsDataItem> getAllDataItems(NxsDataset handler, INode[] nodes) {
        return getAllDataItems( (NxsGroup) handler.getRootGroup(), nodes, 0);
    }

    /**
     * Recursively explore the tree represented by the given array of nodes beginning
     * the exploration at the depth node
     */
    private static List<NxsDataItem> getAllDataItems(NxsGroup entryPoint, INode[] nodes, int depth) {
        List<NxsDataItem> result = new ArrayList<NxsDataItem>();

        if( depth < nodes.length ) {
            INode node = nodes[depth];
            if( depth < nodes.length - 1 ) {
                List<IGroup> groups = entryPoint.getGroupList();

                for( IGroup current : groups ) {
                    INode leaf = ((NxsGroup) current).getNxsPath().getCurrentNode();
                    if( leaf.matchesPartNode(node) ) {
                        result.addAll( getAllDataItems( (NxsGroup) current, nodes, depth + 1) );
                    }
                }
            }
            else {
                List<IDataItem> items = entryPoint.getDataItemList();
                for( IDataItem current : items ) {
                    INode leaf = ((NxsDataItem) current).getPath()
                            .getCurrentNode();
                    if( leaf.matchesPartNode(node) ) {
                        result.add( (NxsDataItem) current );
                    }
                }
            }
        }
        return result;
    }
}
