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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus.dictionary;

import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDataset;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.utils.NxsConstant;
import org.cdma.utilities.configuration.ConfigDataset;

public class NxsLogicalGroup extends LogicalGroup {

    public NxsLogicalGroup(final IDataset dataset, final Key key) {
        super(key, dataset);
    }

    public NxsLogicalGroup(final IDataset dataset, final Key key, final boolean debug) {
        super(key, dataset, debug);
    }

    public NxsLogicalGroup(final LogicalGroup parent, final Key key, final IDataset dataset) {
        super(parent, key, dataset, false);
    }

    public NxsLogicalGroup(final LogicalGroup parent, final Key key, final IDataset dataset, final boolean debug) {
        super(parent, key, dataset, debug);
    }

    @Override
    public ExtendedDictionary findAndReadDictionary() {
        IFactory factory = NxsFactory.getInstance();

        // Detect the key dictionary file and mapping dictionary file
        String keyFile = Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder(factory)
                + NxsLogicalGroup.detectDictionaryFile((NxsDataset) getDataset());

        ExtendedDictionary dictionary = new ExtendedDictionary(factory, keyFile, mapFile);
        try {
            dictionary.readEntries();
        } catch (FileAccessException e) {
            Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
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
    public static String detectDictionaryFile(final NxsDataset dataset) {
        String beamline = "UNKNOWN";
        String model = "UNKNOWN";

        // Get the configuration
        ConfigDataset conf;
        try {
            conf = dataset.getConfiguration();
            // Ask for beamline and datamodel parameters
            beamline = conf.getParameter(NxsConstant.BEAMLINE);
            model = conf.getParameter(NxsConstant.MODEL);
        } catch (NoResultException e) {
            Factory.getLogger().log(Level.WARNING, e.getMessage());
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
