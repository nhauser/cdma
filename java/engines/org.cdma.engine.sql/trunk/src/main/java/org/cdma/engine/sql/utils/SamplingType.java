/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.sql.utils;

import java.text.SimpleDateFormat;

public interface SamplingType {
    /**
     * The DBMS string representation for the date transformation
     */
    public String getSQLRepresentation();

    public String getPattern(SamplingPeriod period);

    /**
     * The DBMS string representation for the date transformation
     * according the given format.
     * 
     * @note only following keys are authorized: [ yyyy, MM, dd, HH, mm, ss, SSS ]
     */
    public String getSQLRepresentation(SimpleDateFormat format);

    /**
     * For the given SamplingPeriod it will return the corresponding sampling format
     * 
     * @param sample
     */
    public SamplingType getType(SamplingPeriod sample);

    /**
     * Return the SQL representation for a sampling on the specified field
     * 
     * @param field
     * @param policy
     * @param name
     * @return
     */
    public String getSamplingSelectClause(String field, SamplingPolicy policy, String name);

    public String getFieldAsStringSelector(String field);

    /**
     * The SamplingPeriod of a date determine which is the precision of the date. More precisely
     * all that is smaller than the mentioned precision will be truncated. The following periods
     * are available:<br/>
     * - MONTH : year and month<br/>
     * - DAY : year, month and day<br/>
     * - HOUR : year, month, day and hour<br/>
     * - MINUTE : year, month, day, hour and minute<br/>
     * - SECOND : year, month, day, hour, minute and seconds<br/>
     * - FRACTION : year, month, day, hour, minute, seconds and fractional<br/>
     */
    public enum SamplingPeriod {
        MONTH(5), DAY(4), HOUR(3), MINUTE(2), SECOND(1), FRACTION(0), NONE(-1);

        private final int mSampling;

        private SamplingPeriod(int sampling) {
            mSampling = sampling;
        }

        public int value() {
            return mSampling;
        }

        public static SamplingPeriod instantiate(int period) {
            SamplingPeriod result;
            switch (period) {
                case 5:
                    result = MONTH;
                    break;
                case 4:
                    result = DAY;
                    break;
                case 3:
                    result = HOUR;
                    break;
                case 2:
                    result = MINUTE;
                    break;
                case 1:
                    result = SECOND;
                    break;
                case 0:
                    result = FRACTION;
                case -1:
                default:
                    result = NONE;
                    break;
            }
            return result;
        }
    }

    /**
     * Return a condition in the WHERE clause for the given field that will be selected
     * according the period and the factor.
     * 
     * @param field
     * @param period
     * @param factor
     * @return a String or null if period or factor are wrong
     */
    public String getDateSampling(String field, SamplingPeriod period, int factor);

    public enum SamplingPolicy {
        NONE(-1), MIN(0), MAX(1), AVERAGE(2);

        private final int mSampling;

        private SamplingPolicy(int sampling) {
            mSampling = sampling;
        }

        public int value() {
            return mSampling;
        }
    }

    public String getSamplingPeriodUnit(SamplingPeriod period);

    public String getPatternPeriodUnit(SamplingPeriod period);
}
