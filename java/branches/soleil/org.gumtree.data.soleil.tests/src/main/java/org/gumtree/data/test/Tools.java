package org.gumtree.data.test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.ISliceIterator;
import org.slf4j.profiler.Profiler;
import org.slf4j.profiler.TimeInstrument;

public class Tools {
	protected static int currentCell = 0;
	protected static Object memory = null;
	protected static Map<String, Long> time = new TreeMap<String, Long>();
	// -------------------------------------------------------------------------------
	// -------------------------------------------------------------------------------
	// TOOLS 
	// -------------------------------------------------------------------------------
	// -------------------------------------------------------------------------------
	static public Object scanArray(Object source, String methodName) {
		Method[] list = Tools.class.getMethods();
		currentCell = 0;
		memory = null;
		for( Method found : list ) {
			if( found.getName().equals(methodName) ) {
				return scanArray(source, found, new ArrayList<Integer>(), 0);
			}
		}
		System.out.println("Method not found: " + methodName + "!!!!!!");
		return null;
	}
	
	static public Object scanArray(Object source, Method job, List<Integer> position, int rank) {
		Object result = null;
		Object item = java.lang.reflect.Array.get(source, 0);
		int length = java.lang.reflect.Array.getLength(source);
		
		if( item.getClass().isArray() && java.lang.reflect.Array.get(item, 0).getClass().isArray() ) {
			Object tmpSrc;
			int size = position.size();
			if( rank >= size ) {
				position.add(0);
			}
			int pos = 0;
			for( int i = 0; i < length && i < 100; i++ ) {
				position.set(rank, pos);
				tmpSrc = java.lang.reflect.Array.get(source, i);
				scanArray( tmpSrc, job, position, rank + 1);
				pos++;
			}
		}
		else {
			try {
				result = job.invoke(null, source, length, position);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		return result;
	}
	
	static public void init( Object array, int length, List<Integer> position) {
		for( int i = 0; i < length; i++ ) {
			java.lang.reflect.Array.set(array, i, currentCell++);
		}
	}
	
	static public String display( Object array, int length, List<Integer> position) {
		if( memory == null )
			memory = new StringBuffer();
		StringBuffer tmp = new StringBuffer();
		((StringBuffer) tmp).append("[");
		int i = 0;
		for( int pos : position) {
			((StringBuffer) tmp).append(pos + (i < position.size() - 1 ? ", " : ""));
			i++;
		}
		((StringBuffer) tmp).append("] = { ");
		
		for( i = 0; i < length; i++ ) {
			((StringBuffer) tmp).append(java.lang.reflect.Array.get(array, i) + ", ");
			if( tmp.length() > 1000 )
				break;
		}
		if( ((StringBuffer) memory).length() < 1000000 ) {
			((StringBuffer) memory).append(tmp.length() > 1000 ? tmp.substring(0, 1000) + ", ... " : tmp );
			((StringBuffer) memory).append(" }\n");
		}
		return ((StringBuffer) memory).toString();
	}
	
	static private String toString( int[] array ) {
		String res = "array = [";
		for( int i = 0; i < array.length; i++ ) {
			if( i != 0 ) {
				res += ", ";
			}
			res += array[i];
		}
		res += "]";
		return res;
	}
	
	public static void contractProfiler(Profiler prof) {
    	Long total = 0l;
    	for( TimeInstrument ti : prof.getCopyOfChildTimeInstruments() ) {
    		if( ! time.containsKey(ti.getName() ) ) {
    			time.put(ti.getName(), ti.elapsedTime());
    		}
    		else {
    			time.put(ti.getName(), time.get(ti.getName()) + ti.elapsedTime());
    		}
    		total += ti.elapsedTime();
    	}
    	if( ! time.containsKey(prof.getName() ) ) {
    		time.put(">>> " +prof.getName(), total);
    	}
    	else {
    		time.put(prof.getName(), time.get(prof.getName()) + total);
    	}
    }
	
	public static void displayProfilers() {
		for( Entry<String, Long> en : Tools.time.entrySet() ) {
			System.out.println( en.getKey() + " = " + (en.getValue() / 1000000000. ) + " s");
		}
	}
	
	
	static protected void displaySlices(IArray array, int rank, String region) throws ShapeNotMatchException, InvalidRangeException {
    	int[] shape = array.getShape();

    	if( shape.length == rank ) {
    		StringBuffer buf = new StringBuffer();
			IArrayIterator iter = array.getIterator();
			for(int size: shape) {
				buf.append(", " + size);
			}
			
			buf.append("]: ");
			int i = 0;
			if( array.getElementType().equals( Integer.TYPE ) ) {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.getIntNext() + ",  ");
				}
			}
			else if( array.getElementType().equals( Short.TYPE ) ) {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.getShortNext() + ",  ");
				}
			}
			else if( array.getElementType().equals( Double.TYPE ) ) {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.getDoubleNext() + ",  ");
				}
			}
			else if( array.getElementType().equals( Long.TYPE ) ) {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.getLongNext() + ",  ");
				}
			}
			else if( array.getElementType().equals( Float.TYPE ) ) {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.getFloatNext() + ",  ");
				}
			}
			else {
				while( iter.hasNext() && i++ < 100 ) {
					buf.append(iter.next() + ",  ");
				}
			}
			System.out.println(region + buf.toString().substring(0, (buf.length() > 1000 ? 1000 : buf.length()) ));
    	}
    	else {
    		int i = 0;
    		ISliceIterator slice = array.getSliceIterator(shape.length - 1);
    		while( slice.hasNext() ) {
    			if( region.isEmpty() )
    				region = "region [";
    			String prefix = region + i;
    			if( slice.getSliceShape().length > rank )  {
    				prefix += ", ";
    			}
    			displaySlices(slice.getArrayNext(), rank, prefix);
    			i++;
    		}
    	}
    }
	
	static void compareArrays(IArray original, IArray copy) throws Exception {
		IArrayIterator iterArray1;
		IArrayIterator iterArray2;
		String buff1;
		String buff2;
		int[] shape1, shape2;
		
		// test the plugin name			
		if( original.getFactoryName() != copy.getFactoryName() ) {
			throw new Exception( "Invalid operation on compared plugin appartenance: original is from " + original.getFactoryName() + " compared to copy from " + copy.getFactoryName());
		}
		
		// test the rank
		if( original.getRank() != copy.getRank() ) {
			throw new Exception( "Invalid operation on compared rank: original rank = " + original.getRank() + " compared to copy rank =  " + copy.getRank());
		}
		
		// test the shape
		shape1 = original.getShape();
		shape2 = copy.getShape();
		if( shape1.length != shape2.length ) {
			throw new Exception( "Invalid operation on compared shape: original rank = " + shape1.length + " compared to copy rank =  " + shape2.length);
		}
		Tools.scanArray(shape1, "display");
		buff1 = Tools.memory.toString();
		Tools.scanArray(shape2, "display");
		buff2 = Tools.memory.toString();
		if(! buff1.equals(buff2) ) {
			throw new Exception( "Invalid operation on compared shape: original shape " + buff1 + " compared to copy shape =  " + buff2);
		}
		
		// test the type
		if( ! original.getElementType().equals(copy.getElementType() ) ) {
			throw new Exception( "Invalid operation on compared type: original type = " + original.getElementType().getName() + " compared to copy = " + original.getElementType().getName());
		}
		
		// test the content
		iterArray1 = original.getIterator();
		iterArray2 = copy.getIterator();
		long i = 0;
		
		if( original.getElementType().equals( Integer.TYPE ) ) {
			int obj1;
			int obj2;
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				obj1 = iterArray1.getIntNext();
				obj2 = iterArray2.getIntNext();
				if( obj1 != obj2 ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		else if( original.getElementType().equals( Short.TYPE ) ) {
			short obj1;
			short obj2;
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				obj1 = iterArray1.getShortNext();
				obj2 = iterArray2.getShortNext();
				if( obj1 != obj2 ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		else if( original.getElementType().equals( Double.TYPE ) ) {
			double obj1;
			double obj2;
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				obj1 = iterArray1.getDoubleNext();
				obj2 = iterArray2.getDoubleNext();
				if( obj1 != obj2 ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		else if( original.getElementType().equals( Long.TYPE ) ) {
			long obj1;
			long obj2;
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				obj1 = iterArray1.getLongNext();
				obj2 = iterArray2.getLongNext();
				if( obj1 != obj2 ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		else if( original.getElementType().equals( Float.TYPE ) ) {
			float obj1;
			float obj2;
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				obj1 = iterArray1.getFloatNext();
				obj2 = iterArray2.getFloatNext();
				if( obj1 != obj2 ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		else {
			while( iterArray1.hasNext() || iterArray2.hasNext() ) {
				Object obj1 = iterArray1.next();
				Object obj2 = iterArray2.next();
				if( ! obj1.equals(obj2) ) {
					throw new Exception( "Invalid operation on compared content: original_cell["+ i + "] = " + obj1 + " compared to copy_cell["+ i + "] =  " + obj2);
				}
				i++;
			}
		}
		
		System.out.println("operation ok! ");
	}
}
