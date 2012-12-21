package org.cdma.utilities.conversion;

import org.cdma.utils.ArrayTools;


/**
 * Tools to convert cells of a String array into an array of another type with same dimension.
 * @author rodriguez
 *
 */
public final class ArrayConverters {
	/**
	 * Will give the right array converter according the given class
	 * @param clazz of expected output array's element
	 * @return a StringArrayConverter instance
	 */
	public static StringArrayConverter detectConverter( Class<?> clazz ) {
		StringArrayConverter result = null;
		if( clazz.equals( Integer.TYPE ) ) {
			result = new StringArrayToIntArray();
		}
		else if( clazz.equals( Double.TYPE ) ) {
			result = new StringArrayToDoubleArray();
		}
		else if( clazz.equals( Short.TYPE ) ) {
			result = new StringArrayToShortArray();
		}
		else if( clazz.equals( Long.TYPE ) ) {
			result = new StringArrayToLongArray();
		}
		else if( clazz.equals( Boolean.TYPE ) ) {
			result = new StringArrayToBoolArray();
		}
		else if( clazz.equals( Byte.TYPE ) ) {
			result = new StringArrayToByteArray();
		}
			
		return result;
	}
	
	/**
	 * StringArrayConverter interface provide a method to convert a String array into an array 
	 * of another type with same dimension.
	 * @author rodriguez
	 *
	 */
	public interface StringArrayConverter {
		/**
		 * Fill the target array with converted cells' content of the source array
		 * @param source String array that will be parsed
		 * @param destination array that will be filled
		 */
		public void convert( String[] source, Object destination );
		
		/**
		 * Convert a single dimensional array of primitives into an array of 
		 * string of the same length.
		 * @param source array of primitive
		 * @return a newly created String[] or null if not a single-dimensional array
		 */
		public String[] convert( Object source );
		
		/**
		 * Return the element's type of the destination array
		 */
		public Class<?> primitiveType();
	}
	
	/**
	 * Convert String[] to int[]
	 */
	public static class StringArrayToIntArray implements StringArrayConverter {
		@Override
		public void convert( String[] source, Object destination ) { 
			int[] out = (int[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Integer.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( int value : (int[]) source ) {
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
		public void convert( String[] source, Object destination ) { 
			double[] out = (double[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Double.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( double value : (double[]) source ) {
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
	 * Convert String[] to short[]
	 */
	public static class StringArrayToShortArray implements StringArrayConverter {
		@Override
		public void convert( String[] source, Object destination ) { 
			short[] out = (short[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Short.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( short value : (short[]) source ) {
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
		public void convert( String[] source, Object destination ) { 
			long[] out = (long[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Long.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( long value : (long[]) source ) {
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
		public void convert( String[] source, Object destination ) { 
			byte[] out = (byte[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Byte.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( byte value : (byte[]) source ) {
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
		public void convert( String[] source, Object destination ) { 
			boolean[] out = (boolean[]) destination;
			
			int cell = 0;
			for( String value : source ) {
				out[cell] = Boolean.valueOf( value );
				cell++;
			}
		}

		@Override
		public String[] convert( Object source ) {
			int[] shape = ArrayTools.detectShape(source);
			
			String[] result = null;
			if( shape.length == 1 ) {
				result = (String[]) java.lang.reflect.Array.newInstance( String.class, shape[0] );
				int index = 0;
				for( boolean value : (boolean[]) source ) {
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
