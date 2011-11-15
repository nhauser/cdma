package org.gumtree.data.soleil.internal;

public enum Beamline {
	ANTARES    ("antares"),
	CONTACQ    ("contacq"),
	DISCO      ("disco"),
	METROLOGIE ("metrologie"),
	SWING      ("swing"),
	UNKNOWN    ("unknown");
    

    private String m_name;

    private Beamline(String type)  { m_name = type; }
    public String getName()        { return m_name; }
}
