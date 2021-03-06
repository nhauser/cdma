package org.gumtree.data.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.soleil.navigation.NxsDataItem;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.gumtree.data.soleil.navigation.NxsDatasetFile;
import org.nexusformat.NexusException;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.NexusFileReader;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathNexus;

public class AttributeFilter {
	/**
	 * Stack all found data items to construct an aggregated NxsDataItem
	 * @param context
	 * @return
	 */
	static public List<NxsDataItem> FilterOnLongName(IContext context) {
		ILogicalGroup group = (ILogicalGroup) context.getCaller();
		IPath path          = context.getPath();
		String addr         = path.toString();
		
		// Extract the subpart corresponding to attribute and value
		String[] attr = addr.substring(addr.lastIndexOf("@") + 1 ).split("=");
		addr = addr.substring(0, addr.lastIndexOf("@"));
		
		NexusNode[] nodes   = PathNexus.splitStringToNode(addr);

		NexusFileReader handler = ((NxsDatasetFile) group.getDataset()).getHandler();

		List<DataItem> list = getAllDataItems(handler, nodes);
		List<NxsDataItem> items = new ArrayList<NxsDataItem>();
		String expected = attr[1].replace("*", ".*");
		for( DataItem item : list ) {
			String value = item.getAttribute(attr[0]);
			if( value.matches(expected) ) {
				items.add( new NxsDataItem( item, (NxsDataset) group.getDataset() ) );
			}
		}
		return items;
	}
	
	/**
	 * Recursively explore the tree represented by the given array of nodes 
	 */
	static protected List<DataItem> getAllDataItems( NexusFileReader handler, NexusNode[] nodes ) {
		try {
			handler.closeAll();
		} catch (NexusException e) {
			e.printStackTrace();
		}
		return getAllDataItems(handler, nodes, 0);
	}
	
	/**
	 * Recursively explore the tree represented by the given array of nodes beginning
	 * the exploration at the depth node
	 */
	static private List<DataItem> getAllDataItems( NexusFileReader handler, NexusNode[] nodes, int depth ) {
		List<DataItem> result = new ArrayList<DataItem>();
		
		if( depth < nodes.length && !handler.isOpenedDataItem() ) {
			try {
				NexusNode node = nodes[depth];
				DataItem item;
				
				List<NexusNode> child = handler.listChildren();
				
				for( NexusNode current : child ) {
					if( current.matchesPartNode(node)) {
						// Open the current node
						handler.openNode(current);
						if( depth < nodes.length - 1 ) {
							result.addAll( getAllDataItems(handler, nodes, depth + 1) );
						}
						// Read the dataset's information
						else if( ! handler.getCurrentPath().getCurrentNode().isGroup() ) {
							item = handler.readDataInfo();
							result.add( item );
						}
						// Close the current node
						handler.closeData();
					}
				}
			} catch (NexusException e) {
				e.printStackTrace();
			}
		}
		return result;
	}
}
