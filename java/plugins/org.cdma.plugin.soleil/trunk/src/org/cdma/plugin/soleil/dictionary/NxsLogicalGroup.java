//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.dictionary;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDataset;
import org.cdma.plugin.soleil.NxsFactory;
import org.cdma.plugin.soleil.navigation.NxsDataset;
import org.cdma.plugin.soleil.utils.NxsConstant;
import org.cdma.utilities.configuration.ConfigDataset;

public class NxsLogicalGroup extends LogicalGroup {

    public NxsLogicalGroup(IDataset dataset, Key key) {
        super(key, dataset);
    }

    public NxsLogicalGroup(IDataset dataset, Key key, boolean debug) {
        super(key, dataset, debug);
    }

    public NxsLogicalGroup(LogicalGroup parent, Key key, IDataset dataset) {
        super(parent, key, dataset, false);
    }

    public NxsLogicalGroup(LogicalGroup parent, Key key, IDataset dataset, boolean debug) {
        super(parent, key, dataset, debug);
    }

    public ExtendedDictionary findAndReadDictionary() {
        IFactory factory = NxsFactory.getInstance();

        // Detect the key dictionary file and mapping dictionary file
        String keyFile = Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder( factory ) + NxsLogicalGroup.detectDictionaryFile( (NxsDataset) getDataset() );
        ExtendedDictionary dictionary = new ExtendedDictionary( factory, keyFile, mapFile);
        try {
            dictionary.readEntries();
        } catch (FileAccessException e) {
            Factory.getLogger().log( Level.SEVERE, e.getMessage(), e);
            dictionary = null;
        }

        return dictionary;
    }
    
    /**
     * According to the current corresponding dataset, this method will try to guess which XML
     * dictionary mapping file should be used
     * 
     * @return
     * @throws FileAccessException
     */
    public static String detectDictionaryFile(NxsDataset dataset) {
        String beamline = "UNKNOWN";
        String model = "UNKNOWN";

        // Get the configuration
        ConfigDataset conf;
        try {
            conf = dataset.getConfiguration();

            // Ask for beamline and datamodel parameters
            beamline = conf.getParameter(NxsConstant.BEAMLINE);
            model = conf.getParameter(NxsConstant.MODEL);
        }
        catch (NoResultException e) {
            Factory.getLogger().log( Level.WARNING, e.getMessage());
        }

        if (beamline != null) {
            beamline = beamline.toLowerCase();
        }
        if (model != null) {
            model = model.toLowerCase();
        }

        // Construct the dictionary file name
        return beamline + "_" + model + ".xml";
    }
}
