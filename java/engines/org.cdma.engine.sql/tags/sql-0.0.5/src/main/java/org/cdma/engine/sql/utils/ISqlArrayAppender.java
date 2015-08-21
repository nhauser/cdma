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
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.sql.utils;

import java.sql.ResultSet;
import java.sql.SQLException;

import org.cdma.engine.sql.internal.DataArray;

public interface ISqlArrayAppender {
	/**
	 * Instantiate a DataArray object according the type of the ResultSet at the
	 * given column index.
	 * 
	 * @param set
	 *            SQL resultset to be analyzed for memory allocation type
	 * @param column
	 *            number to consider for the memory allocation
	 * @param nbRows
	 *            number of rows to allocate
	 * @throws SQLException
	 */
	DataArray<?> allocate(ResultSet set, int column, int nbRows) throws SQLException;
	
	/**
	 * Instantiate a DataArray object according the type of the ResultSet at the
	 * given column index.
	 * 
	 * @param set
	 *            SQL resultset to be analyzed for memory allocation type
	 * @param column
	 *            number to consider for the memory allocation
	 * @param nbRows
	 *            number of rows to allocate
	 * @throws SQLException
	 */
	void append(DataArray<?> array, ResultSet set, int column, int row, int type) throws SQLException;
}
