//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.internal;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

import fr.soleil.database.connection.AbstractDataBaseConnector;
import fr.soleil.database.connection.MySQLDataBaseConnector;
import fr.soleil.database.connection.OracleDataBaseConnector;

public class SqlConnector {
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
        System.out.println("--------------OPEN connection------------------");
        System.out.println("mDriver=" + mDriver);
        System.out.println("mDbScheme=" + mDbScheme);
        System.out.println("mUser=" + mUser);
        System.out.println("mPwd=" + mPwd);
        System.out.println("mRac=" + mRac);
        String hostLabel = "mHost";
        String nameLabel = "mDbName";
        if(mRac) {
            hostLabel = "tnsName";
            nameLabel = "onsConfiguration";
        }
        System.out.println(hostLabel + "=" + mHost);
        System.out.println(nameLabel + "=" + mDbName);
        try {
            if ((mConnection == null) || !mConnection.isValid(0) || mConnection.isClosed()) {
                if (mConnection != null) {
                    try {
                        mConnection.close();
                    } catch (SQLException e) {
                    }
                }

                AbstractDataBaseConnector dbConnector = null;
                if ((mDriver != null) && (mDriver.indexOf("oracle") > -1)) {
                    dbConnector = new OracleDataBaseConnector();
                } else {
                    dbConnector = new MySQLDataBaseConnector();
                }

                if (dbConnector != null) {
                    dbConnector.setSchema(mDbScheme);
                    dbConnector.setUser(mUser);
                    dbConnector.setPassword(mPwd);
                    if (mRac && (dbConnector instanceof OracleDataBaseConnector)) {
                        ((OracleDataBaseConnector) dbConnector).setTnsName(mHost);
                        ((OracleDataBaseConnector) dbConnector).setOnsConfiguration("mDbName");
                    } else {
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
        System.out.println("connection successful");
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
}
