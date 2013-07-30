package org.cdma.engine.hdf.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.Datatype;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.h5.H5Datatype;

import org.cdma.Factory;
import org.cdma.engine.hdf.navigation.HdfAttribute;
import org.cdma.interfaces.IAttribute;

public class HdfObjectUtils {

    @SuppressWarnings("unchecked")
    public static Attribute getAttribute(HObject object, String name) {
        Attribute result = null;
        if (name != null && !name.trim().isEmpty()) {
            List<Attribute> attributes;
            try {
                attributes = object.getMetadata();

                for (Attribute attribute : attributes) {
                    if (name.equals(attribute.getName())) {
                        result = attribute;
                        break;
                    }
                }
            }
            catch (Exception e) {
                Factory.getLogger().severe(e.getMessage());
            }
        }
        return result;
    }

    public static List<IAttribute> getAttributeList(String factoryName, HObject object) {
        List<IAttribute> result = new ArrayList<IAttribute>();
        List<?> attributes;
        if (object != null) {
            try {
                attributes = object.getMetadata();
                for (Object attribute : attributes) {
                    IAttribute attr = new HdfAttribute(factoryName, (Attribute) attribute);
                    result.add(attr);
                }
            }
            catch (Exception e) {
                Factory.getLogger().severe(e.getMessage());
            }
        }
        return result;
    }

    public static boolean hasAttribute(HObject object, String name, String value) {
        boolean result = false;
        if (name != null && value != null) {
            Attribute attribute = HdfObjectUtils.getAttribute(object, name);
            if (attribute != null) {
                if (value.equals(attribute.getValue())) {
                    result = true;
                }
            }
        }
        return result;
    }

    public static boolean removeAttribute(HObject object, IAttribute attributeToRemove) {
        boolean result = true;

        if (attributeToRemove == null) {
            return false;
        }

        try {
            object.removeMetadata(HdfObjectUtils.getAttribute(object, attributeToRemove.getName()));
        }
        catch (Exception e) {
            result = false;
        }

        return result;
    }

    public static int[] convertLongToInt(long[] input) {
        if (input == null) {
            return null;
        }
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (int) input[i];
        }
        return output;
    }

    public static void addStringAttribute(HObject object, String name, String value) {
        long[] dims = { 1 };
        try {
            getMetadataList(object).add(
                    new Attribute(name, new H5Datatype(Datatype.CLASS_STRING), dims, value));
        }
        catch (Exception e) {
            Factory.getLogger().warning(e.getMessage());
        }
    }

    public static void addOneAttribute(HObject object, IAttribute attribute) {
        if (attribute != null) {


            Class<?> type = attribute.getType();
            int datatype = getHdfDataTypeForClass(type);

            long[] dims = { 1 };
            try {
                getMetadataList(object).add(
                        new Attribute(attribute.getName(), new H5Datatype(datatype), dims,
                                attribute.getValue().getStorage()));
            }
            catch (Exception e) {
                Factory.getLogger().log(Level.SEVERE, "Unable to copy addOneAttribute", e);
            }
        }

    }

    public static int getHdfDataTypeForClass(Class<?> type) {
        // Default is STRING
        int datatype = Datatype.CLASS_STRING;

        if ((Byte.TYPE.equals(type)) || Byte.class.equals(type)) {
            datatype = Datatype.CLASS_CHAR;
        }
        else if ((Short.TYPE.equals(type)) || Short.class.equals(type)) {
            datatype = Datatype.CLASS_INTEGER;
        }
        else if ((Integer.TYPE.equals(type)) || Integer.class.equals(type)) {
            datatype = Datatype.CLASS_INTEGER;
        }
        else if ((Long.TYPE.equals(type)) || Long.class.equals(type)) {
            datatype = Datatype.CLASS_FLOAT;
        }
        else if ((Float.TYPE.equals(type)) || Float.class.equals(type)) {
            datatype = Datatype.CLASS_FLOAT;
        }
        return datatype;
    }

    @SuppressWarnings("unchecked")
    public static List<Attribute> getMetadataList(HObject object) {
        List<Attribute> h5AttributeList = new ArrayList<Attribute>();
        try {
            h5AttributeList = object.getMetadata();
        }
        catch (Exception e) {
            Factory.getLogger().log(Level.SEVERE, "Unable to copy getMetadataList", e);
        }
        return h5AttributeList;
    }
}
