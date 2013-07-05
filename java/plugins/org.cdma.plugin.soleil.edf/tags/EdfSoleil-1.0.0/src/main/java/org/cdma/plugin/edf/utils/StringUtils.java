package org.cdma.plugin.edf.utils;

public class StringUtils {

    public static boolean isSameString(String str1, String str2) {
        if (str1 == null) {
            return (str2 == null);
        }
        else {
            return str1.equals(str2);
        }
    }

    public static boolean isSameStringIgnoreCase(String str1, String str2) {
        if (str1 == null) {
            return (str2 == null);
        }
        else {
            return str1.equalsIgnoreCase(str2);
        }
    }

}
