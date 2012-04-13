package org.gumtree.data.test;

import org.gumtree.data.IFactory;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IIndex;
import org.gumtree.data.interfaces.ISliceIterator;
import org.gumtree.data.utils.IArrayUtils;

public class TestArrayManual {
  static public boolean testManuallyCreatedArrays(IFactory factory) throws SecurityException, NoSuchMethodException {
    System.out.println("------------------------------------------------------------------");
    System.out.println("------------------------------------------------------------------");
      System.out.println("Testing manually created arrays:");
      System.out.println("------------------------------------------------------------------");
      boolean result = true;
      
      int[] shape = { 2, 3, 30 };
      int size = 1;
      for(int length : shape ) size *= length;
      Object javaArray = java.lang.reflect.Array.newInstance(int.class, size);
      
      System.out.print("Array created: shape");
      Tools.scanArray(shape, "display");
      System.out.print("--> " + Tools.memory.toString() );
      Tools.scanArray(javaArray, "init");
      Tools.scanArray(javaArray, "display");
    System.out.println("values:\n" + Tools.memory.toString() );
      
    IArray array = factory.createArray(int.class, shape, javaArray);
    System.out.println("CDM Array structure: ");
    System.out.println(array.getIndex());
    IArrayIterator iter = array.getIterator();
    
    IArrayUtils util = array.getArrayUtils();
    util = util.reduce();
    
    ISliceIterator slice;
    IArray arrayslice = null;
    try {
      slice = array.getSliceIterator(2);
      arrayslice = slice.getArrayNext(); 
    } catch (ShapeNotMatchException e) {
      e.printStackTrace();
    } catch (InvalidRangeException e) {
      e.printStackTrace();
    }
    IIndex idx = arrayslice.getIndex();
    Object o = arrayslice.getArrayUtils().copyToNDJavaArray();
    idx.reduce();
    arrayslice.setIndex( idx );
    
    String content1 = "", content2 = "";
    while( iter.hasNext() ) {
      content1 += iter.getObjectNext() + ", ";
    }
    int[] copy = (int[]) array.getArrayUtils().copyTo1DJavaArray();
    for( int value : copy ) {
      content2 += value + ", ";
    }
    
    System.out.println( "Scanning with iterator:   " + content1 );
    System.out.println( "Scanning copyTo1DArray: " + content2 );
    
    Object copyND = array.getArrayUtils().copyToNDJavaArray();
    Tools.scanArray(copyND, "display");
    System.out.println("values:\n" + Tools.memory.toString() );
    String result1 = Tools.memory.toString();
    IArray array2 = factory.createArray(int.class, shape, copyND);
    Object copyND2 = array2.getArrayUtils().copyToNDJavaArray();
    Tools.scanArray(copyND, "display");
    System.out.println("values:\n" + Tools.memory.toString() );
    String result2 = Tools.memory.toString();
    System.out.println("Creation from linear or multidimensional array have equivalent use: " + result1.equals(result2) );
    
      return result;
  }
}
