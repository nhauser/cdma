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
	
	// Type values 
    public enum DataType {
        BOOLEAN  (1),
        SHORT    (2),
        LONG     (3),
        FLOAT    (4),
        DOUBLE   (5),
        USHORT   (6),
        ULONG    (7),
        STRING   (8),
        UNKNOWN  (-1);
        
        private int mName;

        private DataType(int type) { mName = type; }
        public int getName()       { return mName; }
        
        public static DataType ValueOf(int type) {
        	DataType result = null;
        	
        	switch (type) {
			case 1:
				result = BOOLEAN;
				break;
			case 2:
				result = SHORT;
				break;
			case 3:
				result = LONG;
				break;
			case 4:
				result = FLOAT;
				break;
			case 5:
				result = DOUBLE;
				break;
			case 6:
				result = USHORT;
				break;
			case 7:
				result = ULONG;
				break;
			case 8:
				result = STRING;
				break;
			default:
				result = UNKNOWN;
				break;
			}
        	
        	return result;
        }
    }
}
