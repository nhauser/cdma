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
package org.cdma.plugin.mambo.navigation;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingDataset.ArchivingMode;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.BackupException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.mambo.SoleilMamboFactory;
import org.cdma.plugin.mambo.internal.MamboConstants;
import org.cdma.plugin.xml.navigation.XmlDataset;
import org.cdma.plugin.xml.navigation.XmlGroup;
import org.cdma.utilities.configuration.ConfigDataset;
import org.cdma.utilities.configuration.ConfigManager;

import fr.soleil.lib.project.SystemUtils;
import fr.soleil.lib.project.math.ArrayUtils;

public class SoleilMamboDataset implements IDataset {
    private ConfigDataset mConfig;
    private ArchivingDataset mArcDataset;
    private final XmlDataset mXmlDataset;
    private ArchivingMode mArchivingMode;
    private SoleilMamboGroup mPhyRoot;
    private LogicalGroup mLogRoot;
    private boolean mNumDate;
    private final boolean mForceConfig;
    private String beamline = null;

    public SoleilMamboDataset(File file) {
        this(file, true);
    }

    public SoleilMamboDataset(File file, boolean forceConfig) {
        mXmlDataset = new XmlDataset(SoleilMamboFactory.NAME, file);
        mArcDataset = null;
        mArchivingMode = ArchivingMode.HDB;
        mNumDate = true;
        mForceConfig = forceConfig;
    }

    @Override
    public String getFactoryName() {
        return SoleilMamboFactory.NAME;
    }

    @Override
    public void close() throws IOException {
        if (mArcDataset != null) {
            mArcDataset.close();
            mArcDataset = null;
        }
        mXmlDataset.close();
    }

    @Override
    public IGroup getRootGroup() {
        if (mPhyRoot == null) {
            try {
                XmlGroup xmlRoot = (XmlGroup) mXmlDataset.getRootGroup();
                ArchivingGroup arcRoot = (ArchivingGroup) mArcDataset.getRootGroup();
                mPhyRoot = new SoleilMamboGroup(this, null, xmlRoot, arcRoot);
            } catch (BackupException e) {
                e.printStackTrace();
            }
        }
        return mPhyRoot;
    }

    @Override
    public LogicalGroup getLogicalRoot() {
        if (mLogRoot == null) {
            mLogRoot = new LogicalGroup(null, this);
        }
        throw new NotImplementedException();
        // return mLogRoot;
    }

    @Override
    public String getLocation() {
        return mXmlDataset.getLocation();
    }

    @Override
    public String getTitle() {
        return mXmlDataset.getTitle();
    }

    @Override
    public void setLocation(String location) {
        mXmlDataset.setLocation(location);
    }

    @Override
    public void setTitle(String title) {
        mXmlDataset.setTitle(title);
    }

    @Override
    public boolean sync() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public void open() throws IOException {
        initialize();

        if (mArcDataset == null) {
            String user = getUser(mArchivingMode);
            String pwrd = getPassword(mArchivingMode);
            boolean isRac = Boolean.parseBoolean(getRac());
            String schema = getSchema();
            getDbName();
            String driver = getDriver();
            String host = getHost();
            String name = getDbName();
            try {
                mArcDataset = new ArchivingDataset(SoleilMamboFactory.NAME, mArchivingMode, user, pwrd, isRac, schema,
                        name, driver, host, mNumDate);
            } catch (Exception e) {
                mArcDataset = null;
                throw new IOException("Unable to connect to: " + host, e);
            }
        }

        mArcDataset.open();
    }

    /**
     * Open the VC file and parameterize the archiving dataset.
     */
    private void initialize() {
        if (mXmlDataset != null) {
            // Get the XML root group and initialize child
            XmlGroup group = (XmlGroup) mXmlDataset.getRootGroup();

            if (group != null) {
                // Parameterize archiving mode according VC file
                IAttribute attr = group.getAttribute(MamboConstants.ATTRIBUTE_HISTORIC);
                if ((attr != null) && attr.isString()) {
                    boolean historic = Boolean.parseBoolean(attr.getStringValue());
                    mArchivingMode = historic ? ArchivingMode.HDB : ArchivingMode.TDB;
                }
            }
        }
    }

    @Override
    public void save() throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void saveTo(String location) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void save(IContainer container) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public void save(String parentPath, IAttribute attribute) throws WriterException {
        throw new NotImplementedException();
    }

    @Override
    public boolean isOpen() {
        return (mArcDataset != null) && mArcDataset.isOpen();
    }

    @Override
    public long getLastModificationDate() {
        return mXmlDataset.getLastModificationDate();
    }

    public XmlDataset getXmlDataset() {
        return mXmlDataset;
    }

    public ArchivingDataset getArchivingDataset() {
        return mArcDataset;
    }

    /**
     * Are dates expressed as numerical timestamp or as String
     * 
     * @param is true if dates are expressed as timestamps
     */
    public void setNumericalDate(boolean numerical) {
        mNumDate = numerical;
        if (mArcDataset != null) {
            mArcDataset.setNumericalDate(mNumDate);
        }
    }

    /**
     * Are dates expressed as numerical timestamp or as String
     * 
     * @return true if dates are returned as timestamps
     */
    public boolean getNumericalDate() {
        return mNumDate;
    }

    private String getDriver() {
        return getParam("_DRIVER", mArchivingMode, mForceConfig);
    }

    private String getHost() {
        return getParam("_HOST", mArchivingMode, mForceConfig);
    }

    private String getRac() {
        return getParam("_RAC", mArchivingMode, mForceConfig);
    }

    private String getSchema() {
        String schema = getParam("_SCHEMA", mArchivingMode, mForceConfig);
        if (((schema == null) || schema.isEmpty())
                && ((beamline != null) && !beamline.isEmpty() && !beamline.equals("contacq"))) {
            schema = beamline;
        }
        return schema;
    }

    private String getDbName() {
        return getParam("_NAME", mArchivingMode, mForceConfig);
    }

    private String getPassword(ArchivingMode mode) {
        return getParam("_PASSWORD", mode, mForceConfig);
    }

    private String getUser(ArchivingMode mode) {
        return getParam("_USER", mode, mForceConfig);
    }

    /**
     * Search for the value of the given parameter. The order of search is the following:<br/>
     * 1 - system properties,<br/>
     * 2 - system environment,<br/>
     * 3 - cdma config files
     * 
     * @param name to search
     * @param archiving mode
     * @param forceConfig cdma config override system prop
     * @return the value of the given parameter
     */
    private String getParam(String name, ArchivingMode mode, boolean forceConfig) {
        String result = null;

        if (mode != null) {
            // Check in env.
            String fullName = mode.getName() + name;
            result = SystemUtils.getSystemProperty(fullName);

            // Check in config file
            if ((result == null) || forceConfig) {
                ConfigDataset conf = getConfiguration();
                String tmp = null;
                if ((conf != null) && conf.hasParameter(fullName)) {
                    tmp = conf.getParameter(fullName);
                    if ((tmp != null) && !tmp.isEmpty()) {
                        result = tmp;
                    }
                }
            }
        }
        return result;
    }

    /**
     * Get the configuration matching this dataset.
     * 
     * @return
     */
    public ConfigDataset getConfiguration() {
        if (mConfig == null) {
            try {
                beamline = SystemUtils.getSystemProperty(SoleilMamboFactory.BEAMLINE_ENV);
                // If no beamline is defined in system property use the rcm default configuration file
                if ((beamline == null) || beamline.isEmpty()) {
                    mConfig = ConfigManager.getInstance(SoleilMamboFactory.getInstance(),
                            SoleilMamboFactory.CONFIG_FILE).getConfig(this);
                } else if (beamline.equals("contacq")) {
                    // read the test configuration file
                    mConfig = ConfigManager.getInstance(SoleilMamboFactory.getInstance(),
                            "test_" + SoleilMamboFactory.CONFIG_FILE).getConfig(this);
                } else if (beamline.equals("degrad")) {
                    // read the test configuration file
                    mConfig = ConfigManager.getInstance(SoleilMamboFactory.getInstance(),
                            "degrad_" + SoleilMamboFactory.CONFIG_FILE).getConfig(this);
                } else {
                    // read the beamline configuration file
                    mConfig = ConfigManager.getInstance(SoleilMamboFactory.getInstance(),
                            "beamline_" + SoleilMamboFactory.CONFIG_FILE).getConfig(this);
                }
            } catch (NoResultException e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to get configuration!", e);
            }
        }
        return mConfig;
    }

    private static void explore(IGroup group) throws IOException {
        List<IGroup> groups = group.getGroupList();
        for (IGroup iGroup : groups) {
            explore(iGroup);
        }
        List<IDataItem> items = group.getDataItemList();
        for (IDataItem item : items) {
            Factory.getLogger().log(Level.FINE, "Item: ", item.getName());
            Factory.getLogger().log(Level.FINE, "Value: ", ArrayUtils.toString(item.getData().getStorage()));
        }

    }

    public static void main(String[] args) throws Exception {
        System.setProperty("HDB_USER", "hdb");
        System.setProperty("HDB_PASSWORD", "hdb");
        System.setProperty("HDB_SCHEMA", "hdb");
        System.setProperty("HDB_NAME", "TEST11");
        System.setProperty("HDB_HOST", "LUTIN");
        System.setProperty("HDB_RAC", Boolean.toString(false));
        System.setProperty("HDB_DRIVER", "oracle.jdbc.driver.OracleDriver");
        IDataset dataset = new SoleilMamboDataset(new File("/Users/viguier/Work/Projets/AutoBioSAXS/TestGV.vc"), false);
        dataset.open();
        IGroup root = dataset.getRootGroup();
        if (root != null) {
            explore(root);
        }
    }
}
