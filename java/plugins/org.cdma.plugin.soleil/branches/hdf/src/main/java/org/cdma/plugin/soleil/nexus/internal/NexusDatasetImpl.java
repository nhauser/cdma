//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.nexus.internal;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;

import navigation.HdfDataset;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.plugin.soleil.nexus.NxsFactory;

public class NexusDatasetImpl extends HdfDataset {

    // ---------------------------------------------------------
    // Internal class that concretes the abstract NexusDataset
    // ---------------------------------------------------------
    public NexusDatasetImpl(File nexusFile) throws FileAccessException {
        super(NxsFactory.NAME, nexusFile);
        try {
            open();
        }
        catch (IOException e) {
            Factory.getLogger().log(Level.SEVERE, e.getMessage());
        }
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        return new LogicalGroup(null, null, this, false);
    }
}
