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
package org.cdma.engine.archiving.internal.attribute;

import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.utils.DbUtils;
import org.cdma.engine.sql.utils.DbUtils.BaseType;

public class AttributeConnector {
	private SqlDataset dataset;
	private BaseType dbType;
	private String dbName;
	
	public AttributeConnector( SqlDataset dataset, String dbName ) {
		this.dataset = dataset;
		this.dbType  = DbUtils.detectDb(dataset);
		this.dbName  = dbName;
	}
	
	public String getDbName() {
		return dbName;
	}
	
	public BaseType getDbType() {
		return dbType;
	}
	
	public SqlDataset getSqlDataset() {
		return dataset;
	}
	
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append("Location: " + dataset.getLocation());
		result.append("\nDb name: " + dbName );
		result.append("\nDb type: " + dbType );
		return result.toString();
	}
}
