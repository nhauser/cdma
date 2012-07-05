package org.gumtree.data.soleil.internal;

public enum DataModel {
	SCANSERVER   ("scanserver"),
	PASSERELLE   ("passerelle"),
	FLYSCAN      ("flyscan"),
	UNKNOWN      ("unknown");
    

    private String m_name;

    private DataModel(String type)  { m_name = type; }
    public String getName()        { return m_name; }
}
