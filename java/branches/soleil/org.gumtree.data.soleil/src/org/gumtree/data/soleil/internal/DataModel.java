package org.gumtree.data.soleil.internal;

public enum DataModel {
	SCANSERVER   ("scanserver"),
	PASSERELLE   ("passerelle"),
	FLYSCAN      ("flyscan"),
	QUICKEXAFS   ("quickexafs"),
	UNKNOWN      ("unknown");
    

    private String mName;

    private DataModel(String type)  { mName = type; }
    public String getName()        { return mName; }
}
