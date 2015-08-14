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
package org.cdma.engine.sql.navigation;

import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.Path;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.engine.sql.utils.ISqlDataset;
import org.cdma.engine.sql.utils.SqlCdmaCursor;
import org.cdma.exception.CDMAException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.utilities.CDMAExceptionManager;
import org.cdma.utils.Utilities.ModelType;

public class SqlGroup implements IGroup, Cloneable {
    private final String mFactory;
    private SqlGroup mParent;
    private final ISqlDataset mDataset;
    private String[] mDepth;
    private String mName;

    private List<IGroup> mChildGroups;
    private List<IDataItem> mChildItems;
    private SqlCdmaCursor mCursor;

    public SqlGroup(ISqlDataset dataset, String name, ResultSet sql_set) {
        mFactory = dataset.getFactoryName();
        mDataset = dataset;
        mName = name;
        addItems(sql_set);
        mChildGroups = new ArrayList<IGroup>();
    }

    private SqlGroup(SqlGroup group) {
        mFactory = group.mFactory;
        mDataset = group.mDataset;
        mName = group.mName;
        mChildItems = group.mChildItems;
        mChildGroups = new ArrayList<IGroup>();
    }

    public SqlGroup(ISqlDataset dataset, String name, SqlCdmaCursor cursor) {
        mFactory = dataset.getFactoryName();
        mDataset = dataset;
        mName = name;
        mCursor = cursor;
        // childGroups will be instantiated later with cursor data
        mChildGroups = null;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        throw new NotImplementedException();
    }

    @Override
    public void addStringAttribute(String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public IAttribute getAttribute(String name) {
        throw new NotImplementedException();
    }

    @Override
    public List<IAttribute> getAttributeList() {
        throw new NotImplementedException();
    }

    @Override
    public String getLocation() {
        return mDataset.getLocation();
    }

    @Override
    public String getName() {
        String name;
        if (mParent != null) {
            name = mParent.getShortName() + "/" + getShortName();
        } else {
            name = "/" + getShortName();
        }
        return name;
    }

    @Override
    public String getShortName() {
        return mName;
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        throw new NotImplementedException();
    }

    @Override
    public void setName(String name) {
        throw new NotImplementedException();
    }

    @Override
    public void setShortName(String name) {
        mName = name;
    }

    @Override
    public void setParent(IGroup group) {
        if (group instanceof SqlGroup) {
            mParent = (SqlGroup) group;
        }
    }

    @Override
    public long getLastModificationDate() {
        return mDataset != null ? mDataset.getLastModificationDate() : 0;
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }

    @Override
    public void addDataItem(IDataItem item) {
        throw new NotImplementedException();
    }

    @Override
    public Map<String, String> harvestMetadata(String mdStandard) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup getParentGroup() {
        return mParent;
    }

    @Override
    public IGroup getRootGroup() {
        return mDataset.getRootGroup();
    }

    @Override
    public void addOneDimension(IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public void addSubgroup(IGroup group) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem getDataItem(String shortName) {
        IDataItem result = null;
        String name;
        for (IDataItem item : mChildItems) {
            name = item.getShortName();
            if (name.equalsIgnoreCase(shortName)) {
                result = item;
                break;
            }
        }
        return result;
    }

    @Override
    public IDataItem findDataItem(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem getDataItemWithAttribute(String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String attribute) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    public IDimension getDimension(String name) {
        throw new NotImplementedException();
    }

    @Override
    public IContainer getContainer(String shortName) {
        IContainer result = getGroup(shortName);
        if (result == null) {
            result = getDataItem(shortName);
        }
        return result;
    }

    @Override
    public IGroup getGroup(String shortName) {
        IGroup result = null;
        for (IGroup group : mChildGroups) {
            if (group.getShortName().equalsIgnoreCase(shortName)) {
                result = group;
                break;
            }
        }
        return result;
    }

    @Override
    public IGroup getGroupWithAttribute(String attributeName, String value) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItem(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public List<IDataItem> getDataItemList() {
        return mChildItems;
    }

    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    @Override
    public List<IDimension> getDimensionList() {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroup(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IGroup findGroup(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public List<IGroup> getGroupList() {

        if (mChildGroups == null) {
            mChildGroups = new ArrayList<IGroup>();
            try {
                while (mCursor.next()) {
                    try {
                        mChildGroups.add(new SqlGroup(mCursor.getGroup()));
                    } catch (Exception e) {
                        // SILENT CATCH
                        // TDGV Add error management here
                    }
                }

            } catch (SQLException e) {
                Factory.getLogger().log(Level.WARNING, "Unable to initialize group list", e);
                CDMAExceptionManager.notifyHandler(this,
                        new CDMAException("Unable to initialize group list " + e.getMessage()));
            }

        }
        return mChildGroups;
    }

    private void addItems(ResultSet set) {
        IDataItem item;
        if (mChildItems == null) {
            mChildItems = new ArrayList<IDataItem>();
        }
        if (set != null) {
            try {
                ResultSetMetaData meta = set.getMetaData();
                String name;
                int count = meta.getColumnCount();
                for (int col = 1; col <= count; col++) {
                    name = meta.getColumnName(col);

                    item = new SqlDataItem(mFactory, this, name, set, col);
                    mChildItems.add(item);
                }
            } catch (SQLException e) {
                Factory.getLogger().log(Level.WARNING, "Unable to initialize group's children", e);
                CDMAExceptionManager.notifyHandler(this,
                        new CDMAException("Unable to initialize group's children " + e.getMessage()));
            }
        }
    }

    @Override
    public IContainer findContainer(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findContainerByPath(String path) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDataItem(String varName) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDimension(String name) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeGroup(IGroup group) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeGroup(String name) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    @Override
    public void setDictionary(IDictionary dictionary) {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary findDictionary() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isRoot() {
        return mParent == null;
    }

    @Override
    public boolean isEntry() {
        return (mParent != null) && mParent.isRoot();
    }

    @Override
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        throw new NotImplementedException();
    }

    @Override
    public IGroup clone() {
        return new SqlGroup(this);
    }

    public ResultSet executeQuery(String sqlQuery) {
        ResultSet result = null;
        SqlConnector sql_connector = mDataset.getSqlConnector();
        if (sql_connector != null) {
            Connection connection;
            try {
                connection = sql_connector.getConnection();
                Statement statement = connection.createStatement();
                result = statement.executeQuery(sqlQuery);
                statement.close();
            } catch (IOException e) {
                Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
                CDMAExceptionManager.notifyHandler(this, new CDMAException("Cannot open connection " + e.getMessage()));
            } catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, e.getMessage(), e);
                CDMAExceptionManager.notifyHandler(this,
                        new CDMAException("Cannot execute query " + sqlQuery + " " + e.getMessage()));
            }

        }

        return result;
    }

    private String getDataItemListQuery(String[] selectors) {
        String query = "";
        String addon = "";

        if (selectors != null) {
            for (String add : selectors) {
                if ((add != null) && !add.isEmpty()) {
                    addon += add + ", ";
                }
            }
        }

        // Get the tables' names
        if (mDepth.length == 0) {
            query = "select " + addon + "table_name from user_tables";
        }
        // Get the sub-group from the table
        else if (mDepth.length == 1) {
            query = "select " + addon + "row_number() over (order by 1) as row_number, " + mName + ".* FROM " + mName;
        } else {
            query = "select " + addon + "* from (" + mParent.getDataItemListQuery() + ") where row_number=" + mName;
        }

        return query;
    }

    private String getDataItemListQuery() {
        return getDataItemListQuery(null);
    }

}
