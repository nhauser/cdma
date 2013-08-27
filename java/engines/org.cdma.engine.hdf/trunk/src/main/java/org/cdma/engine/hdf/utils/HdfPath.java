package org.cdma.engine.hdf.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.cdma.interfaces.INode;

public class HdfPath {

    public static final String PATH_SEPARATOR = "/";
    protected List<INode> nodes = new ArrayList<INode>();

    public HdfPath(INode[] nodes) {
        this.nodes = Arrays.asList(nodes);
    }

    public HdfPath(INode node) {
        this.nodes.add(node);
    }

    /**
     * Split a string representing a NexusPath to extract each node name
     * 
     * @param path
     */
    public static String[] splitStringPath(String path) {
        if (path.startsWith(HdfPath.PATH_SEPARATOR)) {
            return path.substring(1).split(HdfPath.PATH_SEPARATOR);
        }
        else {
            return path.split(HdfPath.PATH_SEPARATOR);
        }
    }

    public static INode[] splitStringToNode(String sPath) {
        HdfNode[] result = null;
        if (sPath != null) {
            String[] names = splitStringPath(sPath);

            int nbNodes = 0;
            for (String name : names) {
                if (!name.isEmpty()) {
                    nbNodes++;
                }
            }

            if (nbNodes > 0) {
                result = new HdfNode[nbNodes];
                int i = 0;
                for (String name : names) {
                    if (!name.isEmpty()) {
                        result[i] = new HdfNode(name);
                        i++;
                    }
                }
            }
            else {
                result = new HdfNode[0];
            }
        }
        return result;
    }

    public INode[] getNodes() {
        return nodes.toArray(new INode[nodes.size()]);
    }

    @Override
    public String toString() {
        StringBuffer result = new StringBuffer();
        for (INode node : nodes) {
            result.append(node.toString());
            result.append(PATH_SEPARATOR);
        }
        return result.toString();
    }

}
