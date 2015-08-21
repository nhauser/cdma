/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.archiving.internal;

public class Constants {
    // Plug-in attributes
    public static final String ISO_DATE_PATTERN = "yyyy-MM-dd HH:mm:ss.SSS";

    public static final String DATE_FORMAT = "dateFormat"; // format of date to be used for archived attribute
                                                           // extraction
    public static final String START_DATE = "startDate"; // starting date for extraction of an archived attribute
    public static final String END_DATE = "endDate"; // ending date for an extraction of an archived attribute
    public static final String ORIGIN_DATE = "originDate"; // when the archived attribute has been recorded for the
                                                           // first time
    public static final String INTERPRETATION = "interpretation";
    public static final String SAMPLING_TYPE = "samplingType";
    public static final String SAMPLING_FACTOR = "samplingFactor";
    /**
     * Attributes' names that permit to drive the plug-in.
     */
    public static final String[] DATE_ATTRIBUTE = new String[] { START_DATE, END_DATE, ORIGIN_DATE };
    public static final String[] DRIVING_ATTRIBUTE = new String[] { DATE_FORMAT, START_DATE, END_DATE, ORIGIN_DATE,
            SAMPLING_TYPE, SAMPLING_FACTOR };

    public static final String[] SAMPLING_TYPE_LIST = new String[] { SamplingType.NONE_VALUE,
            SamplingType.SECOND_VALUE, SamplingType.MINUTE_VALUE, SamplingType.HOUR_VALUE, SamplingType.DAY_VALUE,
            SamplingType.MONTH_VALUE };

    public class SamplingType {
        private static final String NONE_VALUE = "NONE";
        private static final String SECOND_VALUE = "SECOND";
        private static final String MINUTE_VALUE = "MINUTE";
        private static final String HOUR_VALUE = "HOUR";
        private static final String DAY_VALUE = "DAY";
        private static final String MONTH_VALUE = "MONTH";

    }

    // Constant values
    public enum Interpretation {
        UNKNWON("unknown", -1), SCALAR("scalar", 0), SPECTRUM("spectrum", 1), IMAGE("image", 2);

        private int mType;
        private String mName;

        private Interpretation(final String name, final int type) {
            mName = name;
            mType = type;
        }

        public String getName() {
            return mName;
        }

        public int getType() {
            return mType;
        }

        public static Interpretation ValueOf(final int format) {
            Interpretation result = null;

            switch (format) {
                case 0:
                    result = SCALAR;
                    break;
                case 1:
                    result = SPECTRUM;
                    break;
                case 2:
                    result = IMAGE;
                    break;
            }

            return result;
        }

        public static Interpretation ValueOf(final String format) {
            Interpretation result = null;

            if (format.equalsIgnoreCase(SCALAR.getName())) {
                result = SCALAR;
            } else if (format.equalsIgnoreCase(SPECTRUM.getName())) {
                result = SPECTRUM;
            } else if (format.equalsIgnoreCase(IMAGE.getName())) {
                result = IMAGE;
            }

            return result;
        }
    }

    // Type values
    public enum DataType {
        BOOLEAN(1), SHORT(2), LONG(3), FLOAT(4), DOUBLE(5), USHORT(6), ULONG(7), STRING(8), UNKNOWN(-1);

        private int mName;

        private DataType(final int type) {
            mName = type;
        }

        public int getName() {
            return mName;
        }

        public static DataType ValueOf(final int type) {
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
