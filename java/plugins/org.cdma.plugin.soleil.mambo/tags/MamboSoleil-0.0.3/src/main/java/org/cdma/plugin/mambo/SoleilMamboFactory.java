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
package org.cdma.plugin.mambo;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.sql.Driver;
import java.util.Enumeration;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.arrays.DefaultArray;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.archiving.navigation.ArchivingAttribute;
import org.cdma.engine.archiving.navigation.ArchivingDataItem;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingDataset.ArchivingMode;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.mambo.navigation.SoleilMamboDataset;

import fr.soleil.lib.project.SystemUtils;

public class SoleilMamboFactory implements IFactory {
    public static final String DESC = "Manages a Mambo file to extract archived attributes.";
    public static final String NAME = "MamboSoleil";
    public static final String LABEL = "SOLEIL's Mambo plug-in";
    public static final String API_VERS = "3.2.5";
    public static final String PLUG_VERS = "0.0.1";
    public static final String CONFIG_FILE = "cdma_mambosoleil_config.xml";
    public static final String BEAMLINE_ENV = "BEAMLINE";

    private static SoleilMamboFactory factory;

    static public SoleilMamboFactory getInstance() {
        synchronized (SoleilMamboFactory.class ) {
            if( ! isPluginValid() ) {
                Factory.getManager().unregisterFactory(NAME);
                factory = null;
            }
            else if( factory == null ) {
                factory  = new SoleilMamboFactory();
            }
        }
        return factory;
    }

    @Override
    public IDataset openDataset(URI uri) {
        SoleilMamboDataset dataset = null;
        // Get file path
        String path = uri.getPath();
        File file   = new File(path);

        // Construct dataset
        if( file.exists() ) {
            dataset = new SoleilMamboDataset(file);
            try {
                // Open dataset
                dataset.open();
            } catch (IOException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to open dataset!", e);
            }
        }
        return dataset;
    }

    @Override
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary openDictionary(String filepath)
    throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
        Object storage = java.lang.reflect.Array.newInstance(clazz, shape);
        return this.createArray( clazz, shape, storage );
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        IArray result;
        try {
            result = DefaultArray.instantiateDefaultArray( SoleilMamboFactory.NAME, storage, shape);
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to create array!", e);
        }
        return result;
    }

    @Override
    public IArray createArray(Object storage) {
        IArray result = null;
        try {
            result = DefaultArray.instantiateDefaultArray( SoleilMamboFactory.NAME, storage );
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to create array!", e);
        }
        return result;
    }

    @Override
    public IArray createStringArray(String value) {
        return this.createArray( value );
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArrayNoCopy(Object javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
        IDataItem result = new ArchivingDataItem(NAME, shortName, parent, array);
        return result;
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
        return new ArchivingGroup(NAME, null, null, shortName );
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        ArchivingGroup group = null;
        if( (shortName != null) && !shortName.isEmpty() ) {
            if( parent != null ) {
                if( parent instanceof ArchivingGroup ) {
                    group = new ArchivingGroup(NAME, (ArchivingDataset) parent.getDataset(), (ArchivingGroup) parent, shortName);
                }
            }
            else {
                group = new ArchivingGroup(NAME, null, null, shortName );
            }
        }
        return group;
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        return new LogicalGroup(key, dataset);
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        return new ArchivingAttribute(NAME, name, value);

    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws Exception {
        SoleilMamboDataset dataset = null;
        // Get file path
        String path = uri.getPath();
        File file   = new File(path);

        // Construct dataset
        if( file.exists() ) {
            dataset = new SoleilMamboDataset(file);
        }
        else {
            // TODO create the empty Mambo file
        }

        return dataset;
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IKey createKey(String name) {
        return new Key(this, name);
    }

    @Override
    public Path createPath(String path) {
        return new Path(this, path);
    }

    @Override
    public String getName() {
        return SoleilMamboFactory.NAME;
    }

    @Override
    public String getPluginLabel() {
        return LABEL;
    }

    @Override
    public IDatasource getPluginURIDetector() {
        return SoleilMamboDataSource.getInstance();
    }

    @Override
    public IDictionary createDictionary() {
        throw new NotImplementedException();
    }

    @Override
    public String getPluginVersion() {
        return PLUG_VERS;
    }

    @Override
    public String getCDMAVersion() {
        return API_VERS;
    }

    @Override
    public String getPluginDescription() {
        return SoleilMamboFactory.DESC;
    }

    @Override
    public void processPostRecording() {
        synchronized (SoleilMamboFactory.class ) {
            boolean isValid = SoleilMamboFactory.isPluginValid();
            if( ! isValid ) {
                Factory.getManager().unregisterFactory(SoleilMamboFactory.NAME);
            }
        }
    }

    @Override
    public boolean isLogicalModeAvailable() {
        // No logical mode for this plug-in
        return false;
    }

    static private boolean isPluginValid() {
        boolean isValid = false;

        // Check the plug-in has it's environment or system propserties set
        String hdbEnv = ArchivingMode.HDB.getName() + "_USER";
        String tdbEnv = ArchivingMode.TDB.getName() + "_USER";

        String hdbUsr = SystemUtils.getSystemProperty(hdbEnv);
        String tdbUsr = SystemUtils.getSystemProperty(tdbEnv);
        if(
                ( (hdbUsr != null) && !hdbUsr.isEmpty() ) ||
                ( (tdbUsr != null) && !tdbUsr.isEmpty() )
        ) {
            isValid = true;
        }

        Enumeration<Driver> drivers = java.sql.DriverManager.getDrivers();
        if( (drivers != null) && !drivers.hasMoreElements() ) {
            isValid = false;
        }

        return isValid;
    }
}
