package org.cdma.engine.archiving.internal.attribute;

import org.cdma.engine.archiving.internal.SqlFieldConstants;
import org.cdma.engine.sql.navigation.SqlDataset;

public class Attribute implements Cloneable {
	private AttributePath path;
	private AttributeProperties prop;
	private AttributeConnector dbCon;
	
	public Attribute( SqlDataset dataset, String dbName ) {
		this.prop  = null;
		this.path  = new AttributePath();
		this.dbCon = new AttributeConnector( dataset, dbName);
	}
	
	public Attribute( AttributePath path, AttributeProperties properties, AttributeConnector connector ) {
		this.prop  = properties;
		this.path  = path;
		this.dbCon = connector;
	}
	
	public AttributePath getPath() {
		return path;
	}
	
	public AttributeProperties getProperties() {
		return prop;
	}
	
	public AttributeConnector getDbConnector() {
		return dbCon;
	}
	
	public void setPath( AttributePath path ) {
		this.path = path;
	}
	
	public void setProperties( AttributeProperties prop ) {
		this.prop = prop;
	}
	
	public Attribute clone() {
		AttributePath path = null;
		if( this.path != null ) {
			path = this.path.clone();
		}
		
		AttributeProperties prop = null;
		if( this.prop != null ) {
			prop = this.prop.clone();
		}
		return new Attribute( path, prop, dbCon );
	}
	
	/**
	 * Returns true if the given name is the name of dimension for that attribute 
	 */
	public boolean isDimension(String name) {
		boolean result = false;

		if ((name != null && !name.isEmpty())
				&& (name.equalsIgnoreCase(SqlFieldConstants.ATT_FIELD_DIMX) || 
					name.equalsIgnoreCase(SqlFieldConstants.ATT_FIELD_DIMY) ||
					name.equalsIgnoreCase(SqlFieldConstants.ATT_FIELD_TIME))) {
			result = true;
		}

		return result;
	}
	
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append("connection: " + dbCon.toString() + "\n" );
		result.append("path: " + path + "\n");
		result.append("prop: " + prop );
		return result.toString();
		
		
	}
}
