package org.cdma.plugin.archiving;

import java.io.IOException;
import java.net.URI;
import java.sql.Driver;
import java.util.Enumeration;
import java.util.logging.Level;

import org.cdma.AbstractFactory;
import org.cdma.Factory;
import org.cdma.arrays.DefaultArray;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.engine.archiving.navigation.ArchivingAttribute;
import org.cdma.engine.archiving.navigation.ArchivingDataItem;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingDataset.ArchivingMode;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.CDMAException;
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

public class SoleilArcFactory extends AbstractFactory {

    private static final String DESC = "Attempts to access an archiving database to extract archived attributes.";
    public static final String NAME = "SoleilArchiving";
    public static final String LABEL = "SOLEIL's Archiving plug-in";
    public static final String API_VERS = "3.2.3";
    public static final String PLUG_VERS = "1.0.0";

    

    @Override
    public IDataset openDataset(final URI uri) {
        IDataset dataset = null;
        try {
            dataset = createDatasetInstance(uri);
            if (dataset != null) {
                dataset.open();
            }
        } catch (Exception e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to open dataset!", e);
        }
        return dataset;
    }

    @Override
    public IDictionary openDictionary(final URI uri) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary openDictionary(final String filepath) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArray(final Class<?> clazz, final int[] shape) {
        Object storage = java.lang.reflect.Array.newInstance(clazz, shape);
        return this.createArray(clazz, shape, storage);
    }

    @Override
    public IArray createArray(final Class<?> clazz, final int[] shape, final Object storage) {
        IArray result;
        try {
            result = DefaultArray.instantiateDefaultArray(SoleilArcFactory.NAME, storage, shape);
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to create array!", e);
        }
        return result;
    }

    @Override
    public IArray createArray(final Object storage) {
        IArray result = null;
        try {
            result = DefaultArray.instantiateDefaultArray(SoleilArcFactory.NAME, storage);
        } catch (InvalidArrayTypeException e) {
            result = null;
            Factory.getLogger().log(Level.SEVERE, "Unable to create array!", e);
        }
        return result;
    }

    @Override
    public IArray createStringArray(final String value) {
        return this.createArray(value);
    }

    @Override
    public IArray createDoubleArray(final double[] javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createDoubleArray(final double[] javaArray, final int[] shape) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArrayNoCopy(final Object javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem createDataItem(final IGroup parent, final String shortName, final IArray array)
            throws InvalidArrayTypeException {
        IDataItem result = new ArchivingDataItem(NAME, shortName, parent, array);
        return result;
    }

    @Override
    public IGroup createGroup(final String shortName) throws IOException {
        return new ArchivingGroup(NAME, null, null, shortName);
    }

    @Override
    public IGroup createGroup(final IGroup parent, final String shortName) {
        ArchivingGroup group = null;
        if ((shortName != null) && !shortName.isEmpty()) {
            if (parent != null) {
                if (parent instanceof ArchivingGroup) {
                    group = new ArchivingGroup(NAME, (ArchivingDataset) parent.getDataset(), (ArchivingGroup) parent,
                            shortName);
                }
            } else {
                group = new ArchivingGroup(NAME, null, null, shortName);
            }
        }
        return group;
    }

    @Override
    public LogicalGroup createLogicalGroup(final IDataset dataset, final IKey key) {
        return new LogicalGroup(key, dataset);
    }

    @Override
    public IAttribute createAttribute(final String name, final Object value) {
        return new ArchivingAttribute(NAME, name, value);

    }

    @Override
    public IDataset createDatasetInstance(final URI uri) throws CDMAException {

        ArchivingDataset dataset = null;
        // URI format type jdbc:mysql://dbhost/dbname#hdb;user;passwd;rac;schema
        if (uri != null) {
            String uriString = uri.toString();
            String[] uriSplit = uriString.split("#");
            if ((uriSplit != null) && (uriSplit.length > 0)) {
                String dbDriver = "oracle";
                String dbHost = null;
                String dbName = null;
                ArchivingMode dbMode = ArchivingMode.HDB;
                String dbUser = null;
                String dbPassword = null;
                boolean rac = false;
                String dbScheme = null;
                // Parse first token
                if (uriSplit.length > 0) {
                    String url_db = uriSplit[0];
                    if (url_db.contains("mysql")) {
                        dbDriver = "mysql";
                    }
                    String[] url_split = url_db.split("/");
                    if ((url_split != null) && (url_split.length > 1)) {
                        dbName = url_split[url_split.length - 1];
                        dbHost = url_split[url_split.length - 2];
                    }
                }

                if (uriSplit.length > 1) {
                    String[] uriSplitInfos = uriSplit[1].split(";");

                    // Archiving mode
                    if (uriSplitInfos.length > 0) {
                        try {
                            dbMode = ArchivingMode.valueOf(uriSplitInfos[0]);
                        } catch (Exception e) {
                            dbMode = ArchivingMode.HDB;
                        }
                    }

                    // user
                    if (uriSplitInfos.length > 1) {
                        dbUser = uriSplitInfos[1];
                    }

                    // password
                    if (uriSplitInfos.length > 2) {
                        dbPassword = uriSplitInfos[2];
                    }

                    // rac
                    if (uriSplitInfos.length > 3) {
                        rac = Boolean.parseBoolean(uriSplitInfos[3]);
                    }

                    // scheme
                    if (uriSplitInfos.length > 4) {
                        dbScheme = uriSplitInfos[4];
                    }
                }
                dataset = new ArchivingDataset(NAME, dbMode, dbUser, dbPassword, rac, dbScheme, dbName, dbDriver,
                        dbHost, true);
            }
        }

        return dataset;
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IKey createKey(final String name) {
        return new Key(this, name);
    }

    @Override
    public Path createPath(final String path) {
        return new Path(this, path);
    }

    @Override
    public String getName() {
        return SoleilArcFactory.NAME;
    }

    @Override
    public String getPluginLabel() {
        return LABEL;
    }

    @Override
    public IDatasource getPluginURIDetector() {
        return SoleilArcDataSource.getInstance();
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
        return SoleilArcFactory.DESC;
    }

    @Override
    public void processPostRecording() {
        Enumeration<Driver> drivers = java.sql.DriverManager.getDrivers();
        if ((drivers != null) && !drivers.hasMoreElements()) {
            Factory.getManager().unregisterFactory(SoleilArcFactory.NAME);
        }
    }

    @Override
    public boolean isLogicalModeAvailable() {
        // No logical mode for this plug-in
        return false;
    }
}
