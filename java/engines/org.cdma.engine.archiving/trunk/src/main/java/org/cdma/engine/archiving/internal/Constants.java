package org.cdma.engine.archiving.internal;

public class Constants {
	// Plug-in attributes
	public static final String ISO_DATE_PATTERN = "yyyy-MM-dd HH:mm:ss.SSS";
	
	public static final String DATE_FORMAT = "dateFormat"; // format of date to be used for archived attribute extraction
	public static final String START_DATE  = "startDate";  // starting date for extraction of an archived attribute
	public static final String END_DATE    = "endDate";    // ending date for an extraction of an archived attribute
	public static final String ORIGIN_DATE = "originDate"; // when the archived attribute has been recorded for the first time
	
	public static final String INTERPRETATION = "interpretation";
	
	/**
	 * Attributes' names that permit to drive the plug-in.
	 */
	public static final String[] DATE_ATTRIBUTE    = new String[] { START_DATE, END_DATE, ORIGIN_DATE };
	public static final String[] DRIVING_ATTRIBUTE = new String[] { DATE_FORMAT, START_DATE, END_DATE, ORIGIN_DATE };
	
	
	// Constant values
	public static final String INTERPRETATION_SCALAR   = "scalar";
	public static final String INTERPRETATION_SPECTRUM = "spectrum";
	public static final String INTERPRETATION_IMAGE    = "image";
}
