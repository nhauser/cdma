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
package org.cdma.engine.archiving.internal;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.sql.Blob;
import java.sql.Clob;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Types;
import java.util.HashMap;
import java.util.Map;

import org.cdma.engine.archiving.internal.Constants.DataType;
import org.cdma.engine.archiving.internal.attribute.Attribute;
import org.cdma.engine.sql.internal.ArrayDefaultAppender;
import org.cdma.engine.sql.internal.DataArray;
import org.cdma.engine.sql.utils.ISqlArrayAppender;
import org.cdma.utilities.conversion.ArrayConverters;
import org.cdma.utilities.conversion.ArrayConverters.StringArrayConverter;

public class ArchivingArrayAppender extends ArrayDefaultAppender implements ISqlArrayAppender {
    private final Attribute attribute; // Archiving attribute
    private final Map<Integer, Integer> lengths;
    private final Map<Integer, Boolean> converts;
    private final Map<Integer, Class<?>> clazzz;
    private final Map<Integer, StringArrayConverter> converters;
    private final String separator;

    public ArchivingArrayAppender(Attribute attribute, String separator) {
        this.attribute = attribute;
        this.converts = new HashMap<Integer, Boolean>();
        this.clazzz = new HashMap<Integer, Class<?> > ();
        this.lengths = new HashMap<Integer, Integer>();
        this.converters = new HashMap<Integer, StringArrayConverter>();
        this.separator = separator;
    }

    @Override
    public DataArray<?> allocate(ResultSet set, int column, int nbRows) throws SQLException {
        DataArray<?> array = null;

        // Check current type and expected one
        Class<?> current  = null;
        Class<?> expected = attribute.getProperties().getTypeClass();

        // Get informations from result set
        ResultSetMetaData meta = set.getMetaData();

        // Do not consider the TIME field that can be both in String or TIME
        if( !meta.getColumnName(column).equals(SqlFieldConstants.ATT_FIELD_TIME) ) {
            int curType = meta.getColumnType(column);
            switch (curType) {
                case Types.DATALINK:
                case Types.CHAR:
                case Types.VARCHAR:
                case Types.LONGVARCHAR:
                case Types.NCHAR:
                case Types.LONGNVARCHAR:
                case Types.NVARCHAR: {
                    current = String.class;
                    break;
                }
                case Types.BLOB: {
                    current = Blob.class;
                    break;
                }
                case Types.NCLOB:
                case Types.CLOB: {
                    current = Clob.class;
                    break;
                }
            }
        }

        // Check expected and current type are equals
        if( (current != null) && !current.equals(expected) && !expected.equals(String.class) ) {
            converts.put(column, true);

            // Calculate number of rows
            int rows = nbRows;
            if (nbRows <= 0) {
                rows = 1;
            }

            // Detect expected length of each row
            String sample = set.getString(column);
            int length = sample.split(SqlFieldConstants.CELL_SEPARATOR).length;
            lengths.put(column, length);

            // Get the right type
            DataType type = attribute.getProperties().getType();

            // Instantiate a new DataArray<?>
            switch( type ) {
                case DOUBLE: {
                    // Allocate data
                    array = DataArray.allocate( new double[length], rows);

                    // Get the expected array converter
                    Class<?> clazz = Double.TYPE;
                    clazzz.put(column, clazz);
                    converters.put(column, ArrayConverters.detectConverter(clazz));
                    break;
                }
                case FLOAT: {
                    // Allocate data
                    array = DataArray.allocate( new float[length], rows);

                    // Get the expected array converter
                    Class<?> clazz = Float.TYPE;
                    clazzz.put(column, clazz);
                    converters.put(column, ArrayConverters.detectConverter(clazz));
                    break;
                }
                case ULONG:
                case LONG: {
                    // Allocate data
                    array = DataArray.allocate( new long[length], rows);

                    // Get the expected array converter
                    Class<?> clazz = Long.TYPE;
                    clazzz.put(column, clazz);
                    converters.put(column, ArrayConverters.detectConverter(clazz));
                    break;
                }
                case USHORT:
                case SHORT: {
                    // Allocate data
                    array = DataArray.allocate( new short[length], rows);

                    // Get the expected array converter
                    Class<?> clazz = Short.TYPE;
                    clazzz.put(column, clazz);
                    converters.put(column, ArrayConverters.detectConverter(clazz));
                    break;
                }
                case BOOLEAN:
                    if (current != String.class) {// Allocate data
                        array = DataArray.allocate(new boolean[length], rows);
                        // Get the expected array converter
                        Class<?> clazz = Boolean.TYPE;
                        clazzz.put(column, clazz);
                        converters.put(column, ArrayConverters.detectConverter(clazz));
                    } else {
                        boolean b = true;
                        array = DataArray.allocate(b, rows);

                        // Get the expected array converter
                        Class<?> clazz = Boolean.TYPE;
                        clazzz.put(column, clazz);
                        converters.put(column, ArrayConverters.detectConverter(clazz));

                    }
                    break;
                default: {
                    converts.put(column, false);
                    length = -1;
                    break;
                }
            }
        }

        if( array == null ) {
            array = super.allocate(set, column, nbRows);
        }

        return array;
    }

    @Override
    public void append(DataArray<?> array, ResultSet set, int column, int row, int type) throws SQLException {
        boolean convert = false;
        if( converts.containsKey(column) ) {
            convert = converts.get(column);
        }
        if( convert ) {
            StringArrayConverter converter = converters.get(column);
            if( converter != null ) {
                // Split the String to get a String[]
                String value = getString(set, column, type);

                // Instantiate a convenient array of data
                Object destination = java.lang.reflect.Array.newInstance(clazzz.get(column), lengths.get(column));

                if( value != null ) {
                    String[] stringArray = value.split( separator );
                    converter.convert(stringArray, destination);
                }

                // Get the right type
                DataType wanted = attribute.getProperties().getType();

                // Instantiate a new DataArray<?>
                switch( wanted ) {
                    case BOOLEAN:
                        // Get informations from result set
                        ResultSetMetaData meta = set.getMetaData();

                        int curType = meta.getColumnType(column);
                        if (curType == Types.VARCHAR) {
                            boolean[] dest = (boolean[]) destination;
                            array.setData(dest[0], row);

                        } else {
                            array.setData((boolean[]) destination, row);
                        }

                        break;
                    case DOUBLE:
                        array.setData( (double[]) destination, row);
                        break;
                    case FLOAT:
                        array.setData( (float[]) destination, row);
                        break;
                    case LONG:
                    case ULONG:
                        array.setData( (long[]) destination, row);
                        break;
                    case SHORT:
                    case USHORT:
                        array.setData( (short[]) destination, row);
                        break;
                }
            }
            else {
                super.append(array, set, column, row, type);
            }
        }
        else {
            super.append(array, set, column, row, type);
        }
    }




    private String getString(ResultSet set, int column, int type) throws SQLException {
        String result = null;
        switch( type ) {
            case Types.BLOB: {
                Blob blob = set.getBlob(column);
                if( blob != null ) {
                    // Allocate a byte buffer
                    int length = new Long( blob.length() ).intValue();
                    ByteBuffer buffer = ByteBuffer.allocate(length);

                    try {
                        blob.getBinaryStream().read(buffer.array());
                    } catch (IOException e) {
                        throw new SQLException("Unable to read clob from database!", e);
                    }
                    // Get the blob's values
                    buffer.position(0);
                    result = new String(buffer.array());
                }
                break;
            }
            case Types.NCLOB:
            case Types.CLOB: {
                // Get the clob
                Clob clob = set.getClob(column);

                if( clob != null ) {
                    // Allocate a char buffer
                    int length = new Long( clob.length() ).intValue();
                    CharBuffer buffer = CharBuffer.allocate( length );
                    try {
                        clob.getCharacterStream().read(buffer);
                    } catch (IOException e) {
                        throw new SQLException("Unable to read clob from database!", e);
                    }
                    // Get the clob's values
                    buffer.position(0);
                    result = buffer.toString();
                }
                break;
            }
            default:
                result = set.getString(column);
                break;
        }


        return result;
    }


}
