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
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;

public class SqlConnector {
	private String mUser;
	private String mHost;
	private String mPwd;
	private Connection mConnection;

    private static final String ALTER_SESSION_SET_NLS_NUMERIC_CHARACTERS = "alter session set NLS_NUMERIC_CHARACTERS = \". \"";
    private static final String ALTER_SESSION_SET_NLS_DATE_FORMAT_DD_MM_YYYY_HH24_MI_SS         = "alter session set NLS_DATE_FORMAT = 'DD-MM-YYYY HH24:MI:SS'";
    private static final String ALTER_SESSION_SET_NLS_TIMESTAMP_FORMAT_DD_MM_YYYY_HH24_MI_SS_FF = "alter session set NLS_TIMESTAMP_FORMAT = 'DD-MM-YYYY HH24:MI:SS.FF'";

	public SqlConnector(String host, String user, String password) {
		mHost = host;
		mUser = user;
		mPwd = password;
		mConnection = null;
	}
	
	public Connection getConnection() throws IOException {
		if( mConnection == null ) {
			open();
		}
		return mConnection;
	}

	public Connection open() throws IOException {
		try {
			if (mConnection == null || !mConnection.isValid(0)) {
				if( mConnection != null ) {
					try {
						mConnection.close();
					}
					catch(SQLException e ) {
					}
				}
				mConnection = DriverManager.getConnection(mHost, mUser, mPwd);
				if( mHost.matches( "[^:]*:oracle.*" ) ) {
					if (mConnection != null) {
					    Statement stmt = null;
					    try {
						stmt = mConnection.createStatement();
						stmt.executeQuery(ALTER_SESSION_SET_NLS_NUMERIC_CHARACTERS);
						stmt.executeQuery(ALTER_SESSION_SET_NLS_TIMESTAMP_FORMAT_DD_MM_YYYY_HH24_MI_SS_FF);
						stmt.executeQuery(ALTER_SESSION_SET_NLS_DATE_FORMAT_DD_MM_YYYY_HH24_MI_SS);
					    } finally {
						if (stmt != null && !stmt.isClosed()) {
						    stmt.close();
						}
					    }
					}
				}
			}
		} catch (SQLException e) {
			throw new IOException(e);
		}
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
		if( mConnection != null ) {
			try {
				result = mConnection.isValid(0);
			} catch (SQLException e) {
			}
		}
		return result;
	}
/*
	public Statement createStatement() throws IOException {
		Statement statement = null;
		Connection connection = open();
		if (connection != null) {
			try {
				statement = connection.createStatement();
			} catch (SQLException e) {
				throw new IOException(e);
			}
		}
		return statement;

	}
*/
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
