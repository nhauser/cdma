package org.cdma.utilities.memory;


public class ArrayTools {
	/**
	 * Determine the shape of the given array (raged array aren't considered)
	 * 
	 * @param data on which we want to analyze shape
	 * @return array of integer representing the length of each dimension
	 * @note an empty array is returned in case of scalar data
	 */
	public static int[] detectShape(final Object data) {
    	int[] shape;
    	
        // Check data existence
        if (data == null) {
        	shape = new int[] {};
        }
        else {
	        // Determine rank of array (by parsing data array class name)
	        String sClassName = data.getClass().getName();
	        int iRank = 0;
	        int iIndex = 0;
	        char cChar;
	        while (iIndex < sClassName.length()) {
	            cChar = sClassName.charAt(iIndex);
	            iIndex++;
	            if (cChar == '[') {
	                iRank++;
	            }
	        }

	        // Set dimension rank
	        shape = new int[iRank];

	        // Fill dimension size array
	        Object array = data;
	        for (int i = 0; i < iRank; i++) {
	            shape[i] = java.lang.reflect.Array.getLength(array);
	            array    = java.lang.reflect.Array.get(array, 0);
	        }
        }
        return shape;
    }
	
	/**
	 * Increment a counter by 1, according the shape limitation.
	 * The most varying dimension will be the one with the higher index.
	 * 
	 * @param counter to be incremented
	 * @param shape limitation of the counter
	 */
	public static void incrementCounter(int[] counter, int[] shape) {
		for (int i = counter.length - 1; i >= 0; i--) {
			if (counter[i] + 1 >= shape[i] && i > 0) {
				counter[i] = 0;
			} else {
				counter[i]++;
				return;
			}
		}
	}
}
