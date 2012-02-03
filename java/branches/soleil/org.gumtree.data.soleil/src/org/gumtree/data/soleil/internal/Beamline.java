package org.gumtree.data.soleil.internal;

public enum Beamline {
	ANTARES    ("antares"),
	CONTACQ    ("contacq"),
	DISCO      ("disco"),
	METROLOGIE ("metrologie"),
	SWING      ("swing"),
	SAMBA	   ("samba"),
	UNKNOWN    ("unknown");
    

    private String mName;

    private Beamline(String type)  { mName = type; }
    public String getName()        { return mName; }
}
