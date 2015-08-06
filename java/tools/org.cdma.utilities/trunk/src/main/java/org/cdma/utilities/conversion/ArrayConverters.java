/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utilities.conversion;

import org.cdma.utils.ArrayTools;

/**
 * Tools to convert cells of a String array into an array of another type with same dimension.
 * 
 * @author rodriguez
 */
public final class ArrayConverters {

    private static StringArrayToIntArray stringArrayToIntArray = null;
    private static StringArrayToDoubleArray stringArrayToDoubleArray = null;
    private static StringArrayToFloatArray stringArrayToFloatArray = null;
    private static StringArrayToShortArray stringArrayToShortArray = null;
    private static StringArrayToLongArray stringArrayToLongArray = null;
    private static StringArrayToBoolArray stringArrayToBoolArray = null;
    private static StringArrayToByteArray stringArrayToByteArray = null;
    /**
     * Will give the right array converter according the given class
     * 
     * @param clazz of expected output array's element
     * @return a StringArrayConverter instance
     */
    public static StringArrayConverter detectConverter(final Class<?> clazz) {
        StringArrayConverter result = null;
        if (clazz.equals(Integer.TYPE)) {
            if (stringArrayToIntArray == null) {
                stringArrayToIntArray = new StringArrayToIntArray();
            }
            result = stringArrayToIntArray;
        }

        else if (clazz.equals(Double.TYPE)) {
            if (stringArrayToDoubleArray == null) {
                stringArrayToDoubleArray = new StringArrayToDoubleArray();
            }
            result = stringArrayToDoubleArray;
        }

        else if (clazz.equals(Float.TYPE)) {
            if (stringArrayToFloatArray == null) {
                stringArrayToFloatArray = new StringArrayToFloatArray();
            }
            result = stringArrayToFloatArray;
        }

        else if (clazz.equals(Short.TYPE)) {
            if (stringArrayToShortArray == null) {
                stringArrayToShortArray = new StringArrayToShortArray();
            }
            result = stringArrayToShortArray;
        }

        else if (clazz.equals(Long.TYPE)) {
            if (stringArrayToLongArray == null) {
                stringArrayToLongArray = new StringArrayToLongArray();
            }
            result = stringArrayToLongArray;
        }

        else if (clazz.equals(Boolean.TYPE)) {
            if (stringArrayToBoolArray == null) {
                stringArrayToBoolArray = new StringArrayToBoolArray();
            }
            result = stringArrayToBoolArray;
        }

        else if (clazz.equals(Byte.TYPE)) {
            if (stringArrayToByteArray == null) {
                stringArrayToByteArray = new StringArrayToByteArray();
            }
            result = stringArrayToByteArray;
        }
        return result;
    }

    /**
     * StringArrayConverter interface provide a method to convert a String array into an array
     * of another type with same dimension.
     * 
     * @author rodriguez
     */
    public interface StringArrayConverter {
        /**
         * * Fill the target array with converted cells' content of the source array * * @param source String array that
         * will be parsed * @param destination array that will be filled
         */
        public void convert(String[] source, Object destination);

        /**
         * * Convert a single dimensional array of primitives into an array of * string of the same length. * * @param
         * source array of primitive * @return a newly created String[] or null if not a single-dimensional array
         */
        public String[] convert(Object source);

        /** * Return the element's type of the destination array */
        public Class<?> primitiveType();
    }

    /**
     * Convert String[] to int[]
     */
    public static class StringArrayToIntArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            int[] out = (int[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Double.valueOf(value).intValue();
                cell++;
            }

        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (int value : (int[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }
            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Integer.TYPE;
        }
    }

    /**
     * Convert String[] to double[]
     */
    public static class StringArrayToDoubleArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            double[] out = (double[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Double.valueOf(value).doubleValue();
                cell++;
            }
        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (double value : (double[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }

            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Double.TYPE;
        }

    }

    /**
     * Convert String[] to double[]
     */
    public static class StringArrayToFloatArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            float[] out = (float[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Double.valueOf(value).floatValue();
                cell++;
            }

        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (float value : (float[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }
            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Float.TYPE;
        }
    }

    /**
     * Convert String[] to short[]
     */
    public static class StringArrayToShortArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            short[] out = (short[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Double.valueOf(value).shortValue();
                cell++;
            }
        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (short value : (short[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }

            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Short.TYPE;
        }

    }

    /**
     * Convert String[] to long[]
     */
    public static class StringArrayToLongArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            long[] out = (long[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Double.valueOf(value).longValue();
                cell++;
            }

        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (long value : (long[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }
            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Long.TYPE;
        }
    }

    /**
     * Convert String[] to byte[]
     */
    public static class StringArrayToByteArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            byte[] out = (byte[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = Byte.valueOf(value);
                cell++;
            }
        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (byte value : (byte[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }

            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Byte.TYPE;
        }

    }

    /**
     * Convert String[] to bool[]
     */
    public static class StringArrayToBoolArray implements StringArrayConverter {
        @Override
        public void convert(final String[] source, final Object destination) {
            boolean[] out = (boolean[]) destination;

            int cell = 0;
            for (String value : source) {
                out[cell] = convertToBoolValue(value);
                cell++;
            }

        }

        private boolean convertToBoolValue(String value) {
            boolean boolValue = false;

            // Check with numerical first 0 or 1
            try {
                Double dbValue = Double.valueOf(value);
                if (dbValue.intValue() == 1) {
                    boolValue = true;
                }
            } catch (NumberFormatException e) {
                // else test with true false
                boolValue = Boolean.valueOf(value);
            }

            return boolValue;
        }

        @Override
        public String[] convert(final Object source) {
            int[] shape = ArrayTools.detectShape(source);

            String[] result = null;
            if (shape.length == 1) {
                result = (String[]) java.lang.reflect.Array.newInstance(String.class, shape[0]);
                int index = 0;
                for (boolean value : (boolean[]) source) {
                    result[index] = String.valueOf(value);
                    index++;
                }

            }

            return result;
        }

        @Override
        public Class<?> primitiveType() {
            return Boolean.TYPE;
        }

    }

}