package org.gumtree.data.test;

import java.io.IOException;

import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.util.CubeData;

public class TestCubeData {

	
	public static void testCube( IDataItem item ) throws Exception {
        IArray array = item.getData();
        array.setIndex(array.getIndex().reduce());
        System.out.println("Constructing cube");
        long st = System.currentTimeMillis();
        CubeData cube = new CubeData(array, 2, 30000000);
        System.out.println("Constructing cube >> done : " + ( System.currentTimeMillis() - st ) / (float) 1000 + "s");
        System.out.println("Get data");
        IArray data;
        int[] shape;
		try {
			shape = item.getData().getShape();
		} catch (IOException e1) {
			shape = null;
		}
        Object copy;
        for( int i = 0; i < shape[0]; i++ ) {
        	System.out.println("-------------------------");
	        for( int j = 0; j < shape[1]; j++ ) {
	        	System.out.println("Get data >> {" + i + ", " + j + "}" );
	        	data = cube.getData( new int[] {i, j});
	        	copy = data.getArrayUtils().copyTo1DJavaArray();
	        	Tools.scanArray(copy, "display");
				System.out.println("values:\n" + Tools.memory.toString() );
	        	System.out.println( data.getObject(data.getIndex()));
	        	if( j > 50 )
	        		break;
	        }
        }
	}
}
