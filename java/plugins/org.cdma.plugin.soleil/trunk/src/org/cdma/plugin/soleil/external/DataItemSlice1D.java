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
import org.cdma.exception.InvalidRangeException;
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
 * Stack all found data items to construct an aggregated IDataItem
 * then returns the first slice of that stack
 * @param context @return
 */
public final class DataItemSlice1D implements IPluginMethod {
    
    @Override
    public void execute(Context context) throws CDMAException {
        List<IContainer> result = new ArrayList<IContainer>();
        IDataItem item = getFirstSlice(context);
        if( item != null ) {
            result.add( item );
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
        Path path = solvers.get( solvers.size() - 2 ).getPath();
        String addr = path.toString();

        // Extract the subpart corresponding to attribute and value
        String[] attr = addr.substring(addr.lastIndexOf('@') + 1 ).split("=");
        addr = addr.substring(0, addr.lastIndexOf('@'));

        NexusNode[] nodes   = PathNexus.splitStringToNode(addr);

        IDataItem item = null;
        List<NxsDataItem> list = getAllDataItems((NxsDataset) group.getDataset(), nodes);
        String expected = attr[1].replace("*", ".*");
        for (NxsDataItem curItem : list) {
            String value = curItem.getAttribute(attr[0]).getStringValue();
            if( value.matches(expected) ) {
                item = curItem;
                try {
                    while (item.getRank() > 1) {
                        item = item.getSlice(0, 0);
                    }
                }
                catch (InvalidRangeException e) {}
                break;
            }
        }
        return item;
    }

    /**
     * Recursively explore the tree represented by the given array of nodes 
     */
    protected static List<NxsDataItem> getAllDataItems( NxsDataset handler, NexusNode[] nodes ) {
        return getAllDataItems( (NxsGroup) handler.getRootGroup(), nodes, 0);
    }

    /**
     * Recursively explore the tree represented by the given array of nodes beginning
     * the exploration at the depth node
     */
    private static List<NxsDataItem> getAllDataItems( NxsGroup entryPoint, NexusNode[] nodes, int depth ) {
        List<NxsDataItem> result = new ArrayList<NxsDataItem>();

        if( depth < nodes.length ) {
            NexusNode node = nodes[depth];
            if( depth < nodes.length - 1 ) {
                List<IGroup> groups = entryPoint.getGroupList();

                for( IGroup current : groups ) {
                    NexusNode leaf = ((NxsGroup) current).getPathNexus().getCurrentNode();
                    if( leaf.matchesPartNode(node) ) {
                        result.addAll( getAllDataItems( (NxsGroup) current, nodes, depth + 1) );
                    }
                }
            }
            else {
                List<IDataItem> items = entryPoint.getDataItemList();
                for( IDataItem current : items ) {
                    NexusNode leaf = ((NxsDataItem) current).getNexusItems()[0].getPath().getCurrentNode(); 
                    if( leaf.matchesPartNode(node) ) {
                        result.add( (NxsDataItem) current );
                    }
                }
            }
        }
        return result;
    }

}
