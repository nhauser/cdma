package org.gumtree.data.soleil.internal;

import org.gumtree.data.interfaces.IDataset;

public final class ConfigParameterStatic implements ConfigParameter {
	private String         mName;     // Parameter name
	private String         mValue;    // Parameter value

	ConfigParameterStatic(String name, String value) {
		mName = name;
		mValue = value;
		
	}
	
	public boolean matches(IDataset dataset) {
		return true;
	}
	
	public String getValue(IDataset dataset) {
		return mValue;
	}
	
	public String getName() {
	    return mName;
	}
}

