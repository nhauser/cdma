package org.cdma.engine.archiving.internal;

import org.cdma.engine.sql.array.SqlArray;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IArrayIterator;
import org.cdma.utilities.conversion.ArrayConverters;
import org.cdma.utilities.conversion.ArrayConverters.StringArrayConverter;
import org.cdma.utilities.performance.PostTreatment;

/**
 * This class aims to asynchronously convert an IArray of String into an array of primitives.
 * Each String contained in the source array will be split according the given separator. We will
 * obtain the a String[] from each cell of the source array. Therefore, each String[] will be parsed
 * and converted into a primitive array that will be set in the targeted array. 
 * 
 * @author rodriguez
 *
 */
public class ConvertStringArray implements PostTreatment {

	private Class<?> clazz;
	private SqlArray source;
	private IArray   target;
	private Object   output;
	private String   separator;

	public ConvertStringArray( SqlArray source, IArray target, String separator ) {
		this.separator = separator;
		this.target = target;
		this.source = source;
		this.output = target.getStorage();
		this.clazz  = target.getElementType();
		target.lock();
	}
	
	@Override
	public String getName() {
		return "String array convertion";
	}
	
	@Override
	public void process() {
		// Convert source
		if( clazz != null ) {
			// Get the expected array converter
			StringArrayConverter converter = ArrayConverters.detectConverter(clazz);
			
			// Get an iterator on each cell of the array
			IArrayIterator iterator = source.getIterator();
			String tmpStr;
			Object tmpOut;
			String[] stringArray;

			// For each cell
			while( iterator.hasNext() ) {
				// Get the next string that corresponds to spectrum of values
				tmpStr = (String) iterator.next();
				if( tmpStr != null ) {
					// Split the String to get a String[]
					stringArray = tmpStr.split( separator );

					// Initialize the output slab from the targeted array storage
					tmpOut = output;
					
					// Get the slab the iterator is currently targeting at
					for( int pos : iterator.getCounter() ) {
						tmpOut = java.lang.reflect.Array.get(tmpOut, pos);
					}
					
					// Convert the slab and set it in output array
					converter.convert(stringArray, tmpOut);
				}
			}
		}
		
		target.unlock();
	}

}
