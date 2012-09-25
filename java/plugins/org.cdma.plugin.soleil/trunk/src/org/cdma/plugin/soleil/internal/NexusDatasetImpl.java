package org.cdma.plugin.soleil.internal;

import java.io.File;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.nexus.navigation.NexusDataset;
import org.cdma.exception.FileAccessException;
import org.cdma.plugin.soleil.NxsFactory;

public class NexusDatasetImpl extends NexusDataset {

    // ---------------------------------------------------------
    // Internal class that concretes the abstract NexusDataset
    // ---------------------------------------------------------
    public NexusDatasetImpl(File nexusFile, boolean resetBuffer) throws FileAccessException {
        super(NxsFactory.NAME, nexusFile, resetBuffer);
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup(null, null, this, false);
    }

}
