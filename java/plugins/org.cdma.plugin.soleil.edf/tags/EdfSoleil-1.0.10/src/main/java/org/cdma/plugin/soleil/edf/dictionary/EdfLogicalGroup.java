/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.edf.dictionary;

import java.io.IOException;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.cdma.plugin.soleil.edf.navigation.EdfDataset;

public class EdfLogicalGroup extends LogicalGroup {

    private static final String FACILITY = "Facility";

    public EdfLogicalGroup(IDataset dataset, Key key) {
        super(key, dataset);
    }

    public EdfLogicalGroup(IDataset dataset, Key key, boolean debug) {
        super(key, dataset, debug);
    }

    public EdfLogicalGroup(LogicalGroup parent, Key key, IDataset dataset) {
        super(parent, key, dataset, false);
    }

    public EdfLogicalGroup(LogicalGroup parent, Key key, IDataset dataset, boolean debug) {
        super(parent, key, dataset, debug);
    }

    /**
     * According to the current corresponding dataset, this method will try to guess which XML
     * dictionary mapping file should be used
     *
     * @return
     * @throws FileAccessException
     */
    public static String detectDictionaryFile(EdfDataset dataset) {
        String dictionary = "default.xml";
        if (dataset.getRootGroup() != null) {
            IGroup root = dataset.getRootGroup();
            if (root != null) {
                IDataItem facilityItem = root.getDataItem(FACILITY);
                if (facilityItem != null) {
                    try {
                        dictionary = facilityItem.readScalarString() + ".xml";
                    } catch (IOException e) {
                        Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
                    }
                }
            }
        }
        return dictionary;
    }

    @Override
    public ExtendedDictionary findAndReadDictionary() {
        IFactory factory = EdfFactory.getInstance();

        // Detect the key dictionary file and mapping dictionary file
        String keyFile = Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder(factory)
                + EdfLogicalGroup.detectDictionaryFile((EdfDataset) getDataset());
        ExtendedDictionary dictionary = new ExtendedDictionary(factory, keyFile, mapFile);
        try {
            dictionary.readEntries();
        } catch (FileAccessException e) {
            Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
            dictionary = null;
        }

        return dictionary;
    }
}
