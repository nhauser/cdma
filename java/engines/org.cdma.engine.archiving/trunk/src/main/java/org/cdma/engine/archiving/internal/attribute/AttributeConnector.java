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
}
