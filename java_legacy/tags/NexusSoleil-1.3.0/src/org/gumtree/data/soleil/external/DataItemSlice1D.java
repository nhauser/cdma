package org.gumtree.data.soleil.external;

import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.soleil.navigation.NxsDataItem;
import org.gumtree.data.soleil.navigation.NxsDatasetFile;
import org.nexusformat.NexusException;

import fr.soleil.nexus4tango.DataItem;
import fr.soleil.nexus4tango.NexusFileReader;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathNexus;

public class DataItemSlice1D {
	public IDataItem getFirstSlice(IContext context) {
		ILogicalGroup group = (ILogicalGroup) context.getCaller();
		IPath path          = context.getPath();
		String addr         = path.toString();
		
		// Extract the subpart corresponding to attribute and value
		String[] attr = addr.substring(addr.lastIndexOf("@") + 1 ).split("=");
		addr = addr.substring(0, addr.lastIndexOf("@"));
		
		NexusNode[] nodes   = PathNexus.splitStringToNode(addr);

		NexusFileReader handler = ((NxsDatasetFile) group.getDataset()).getHandler();
		IDataItem item = null;
		List<DataItem> list = getAllDataSets(handler, nodes);
		List<NxsDataItem> items = new ArrayList<NxsDataItem>();
		String expected = attr[1].replace("*", ".*");
		for (DataItem cur_item : list) {
			String value = cur_item.getAttribute(attr[0]);
			if( value.matches(expected) ) {
				item = new NxsDataItem(cur_item, (NxsDatasetFile) group.getDataset());
				try {
					while (item.getRank() > 1) {
						item = item.getSlice(0, 0);
					}
				}
				catch (InvalidRangeException e) {
					e.printStackTrace();
				}
				break;
			}
		}
		return item;
	}
		
	/**
	 * Recursively explore the tree represented by the given array of nodes
	 */
	static protected List<DataItem> getAllDataSets(NexusFileReader handler, NexusNode[] nodes) {
		try {
			handler.closeAll();
		}
		catch (NexusException e) {
			e.printStackTrace();
		}
		return getAllDataSets(handler, nodes, 0);
	}

	/**
	 * Recursively explore the tree represented by the given array of nodes beginning the
	 * exploration at the depth node
	 */
	static private List<DataItem> getAllDataSets(NexusFileReader handler, NexusNode[] nodes,
			int depth) {
		List<DataItem> result = new ArrayList<DataItem>();

		if (depth < nodes.length && !handler.isOpenedDataItem()) {
			try {
				NexusNode node = nodes[depth];
				DataItem item;

				List<NexusNode> child = handler.listChildren();

				for (NexusNode current : child) {
					if (current.matchesPartNode(node)) {
						// Open the current node
						handler.openNode(current);
						if (depth < nodes.length - 1) {
							result.addAll(getAllDataSets(handler, nodes, depth + 1));
						}
						// Read the dataset's information
						else if (!handler.getCurrentPath().getCurrentNode().isGroup()) {
							item = handler.readDataInfo();
							result.add(item);
						}
						// Close the current node
						handler.closeData();
					}
				}
			}
			catch (NexusException e) {
				e.printStackTrace();
			}
		}
		return result;
	}
}
