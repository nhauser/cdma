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
package org.cdma.engine.sql.internal;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

import org.cdma.engine.sql.navigation.SqlDataset;

import fr.soleil.database.connection.AbstractDataBaseConnector;
import fr.soleil.database.connection.MySQLDataBaseConnector;
import fr.soleil.database.connection.OracleDataBaseConnector;

public class SqlConnector {
    private static final String ORACLE_IDENTIFIER = "oracle";
    private final String mUser;
    private final String mHost;
    private final String mPwd;
    private final boolean mRac;
    private final String mDriver;
    private final String mDbName;
    private final String mDbScheme;
    private Connection mConnection;

    private static final String ALTER_SESSION_SET_NLS_NUMERIC_CHARACTERS = "alter session set NLS_NUMERIC_CHARACTERS = \". \"";
    private static final String ALTER_SESSION_SET_NLS_DATE_FORMAT_DD_MM_YYYY_HH24_MI_SS = "alter session set NLS_DATE_FORMAT = 'DD-MM-YYYY HH24:MI:SS'";
    private static final String ALTER_SESSION_SET_NLS_TIMESTAMP_FORMAT_DD_MM_YYYY_HH24_MI_SS_FF = "alter session set NLS_TIMESTAMP_FORMAT = 'DD-MM-YYYY HH24:MI:SS.FF'";

    public SqlConnector(String host, String user, String password, String driver, String dbName, String dbScheme,
            boolean rac) {
        mHost = host;
        mUser = user;
        mPwd = password;
        mDriver = driver;
        mDbName = dbName;
        mDbScheme = dbScheme;
        mRac = rac;
        mConnection = null;
    }

    public Connection getConnection() throws IOException {
        if (mConnection == null) {
            open();
        }
        return mConnection;
    }

    public Connection open() throws IOException {
//        System.out.println("--------------OPEN connection------------------");
//        System.out.println("mDriver=" + mDriver);
//        System.out.println("mDbScheme=" + mDbScheme);
//        System.out.println("mUser=" + mUser);
//        System.out.println("mPwd=" + mPwd);
//        System.out.println("mRac=" + mRac);

//        String hostLabel = "mHost";
//        String nameLabel = "mDbName";
//        if (mRac) {
//            hostLabel = "tnsName";
//            nameLabel = "onsConfiguration";
//        }
//        System.out.println(hostLabel + "=" + mHost);
//        System.out.println(nameLabel + "=" + mDbName);
        try {
            if ((mConnection == null) || !mConnection.isValid(0) || mConnection.isClosed()) {
                if (mConnection != null) {
                    try {
                        mConnection.close();
                    } catch (SQLException e) {
                    }
                }

                AbstractDataBaseConnector dbConnector = null;
                if ((mDriver != null) && (mDriver.contains(ORACLE_IDENTIFIER))) {
                    dbConnector = new OracleDataBaseConnector();
                } else {
                    dbConnector = new MySQLDataBaseConnector();
                }

                if (dbConnector != null) {

                    dbConnector.setUser(mUser);
                    dbConnector.setPassword(mPwd);
                    if (mRac && (dbConnector instanceof OracleDataBaseConnector)) {
                        ((OracleDataBaseConnector) dbConnector).setTnsName(mHost);
                        ((OracleDataBaseConnector) dbConnector).setOnsConfiguration(mDbName);
                        ((OracleDataBaseConnector) dbConnector).setRac(mRac);
                    } else {
                        dbConnector.setSchema(mDbScheme);
                        dbConnector.setHost(mHost);
                        dbConnector.setName(mDbName);
                    }

                    dbConnector.connect();
                    mConnection = dbConnector.getConnection();

                    if (mConnection != null) {
                        Statement stmt = null;
                        try {
                            stmt = mConnection.createStatement();
                            stmt.executeQuery(ALTER_SESSION_SET_NLS_NUMERIC_CHARACTERS);
                            stmt.executeQuery(ALTER_SESSION_SET_NLS_TIMESTAMP_FORMAT_DD_MM_YYYY_HH24_MI_SS_FF);
                            stmt.executeQuery(ALTER_SESSION_SET_NLS_DATE_FORMAT_DD_MM_YYYY_HH24_MI_SS);
                        } finally {
                            if ((stmt != null) && !stmt.isClosed()) {
                                stmt.close();
                            }
                        }
                    }
                }
            }
        } catch (SQLException e) {
            throw new IOException(e);
        }
//        System.out.println("connection successful");
        return mConnection;
    }

    public void close() throws IOException {
        if (mConnection != null) {
            try {
                mConnection.close();
            } catch (SQLException e) {
                throw new IOException(e);
            }
        }
    }

    public String getHost() {
        return mHost;
    }

    public String getDriver() {
        return mDriver;
    }

    public boolean isOpen() {
        boolean result = false;
        if (mConnection != null) {
            try {
                result = mConnection.isValid(0);
            } catch (SQLException e) {
            }
        }
        return result;
    }

    public PreparedStatement prepareStatement(String sql) throws IOException {
        PreparedStatement statement = null;
        Connection connection = open();
        if (connection != null) {
            try {
                statement = connection.prepareStatement(sql);
            } catch (SQLException e) {
                throw new IOException(e);
            }
        }
        return statement;
    }

    public static void main(String args[]) {
        // CHANGE THIS VALUES
        boolean useRac = false;
        boolean test = true;

        // Values for RCM
        String factoryName = "this";
        String host = "erato.rcm";
        String user = "hdbarchiver";
        String passwd = "hdbarchiver";
        String driver = "jdbc:oracle:thin";
        String dbName = "hdb";
        String dbScheme = "hdb";

        if (test) {
            host = "LUTIN";// "erato.rcm";
            user = "HDB"; // "hdbarchiver";
            passwd = "HDB";// "hdbarchiver";
            driver = "jdbc:oracle:thin";
            dbName = "TEST11";
            dbScheme = "";
        }

        if (useRac) {
            host = "(DESCRIPTION = (ADDRESS_LIST=(LOAD_BALANCE=on)(ADDRESS = (PROTOCOL = TCP)(HOST = calliope-vip.rcm) (PORT = 1521)) (ADDRESS = (PROTOCOL = TCP)(HOST = euterpe-vip.rcm)(PORT = 1521))) (CONNECT_DATA = (SERVICE_NAME = HDB)))"; // "erato.rcm";
            dbName = "thalie:6200,euterpe:6200,calliope:6200";
        }
        SqlDataset ds = new SqlDataset(factoryName, host, user, passwd, driver, dbName, dbScheme, useRac);

        try {
            ds.open();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
