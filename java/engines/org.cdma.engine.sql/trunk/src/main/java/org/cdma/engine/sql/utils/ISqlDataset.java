package org.cdma.engine.sql.utils;

import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.interfaces.IDataset;

public interface ISqlDataset extends IDataset {

	public SqlConnector getSqlConnector();
}
