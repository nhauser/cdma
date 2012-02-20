package org.gumtree.data.soleil.internal;

import org.gumtree.data.interfaces.IDataset;

public interface ConfigParameter {
	public String getValue(IDataset dataset);
	public String getName();
	
	public enum CriterionType {
		EXIST     ("exist"),
		NAME      ("name"),
		VALUE     ("value"),
		CONSTANT  ("constant"),
		EQUAL     ("equal"),
		//NOT_EQUAL ("not_equal"),
		NONE      ("");
	    
	    private String mName;

	    private CriterionType(String type) { mName = type; }
	    public String getName()            { return mName; }
	}
	
	public enum CriterionValue {
		TRUE     ("true"),
		FALSE    ("false"),
		NONE     ("");
	    
	    private String mName;

	    private CriterionValue(String type) { mName = type; }
	    public String getName()             { return mName; }
	}
}
