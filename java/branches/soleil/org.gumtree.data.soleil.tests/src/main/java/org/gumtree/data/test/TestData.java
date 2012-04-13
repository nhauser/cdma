package org.gumtree.data.test;

import java.io.IOException;
import java.util.List;

import org.gumtree.data.Factory;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IArray;
import org.gumtree.data.interfaces.IArrayIterator;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IKey;

public class TestData {
   public static void testData(ILogicalGroup root) throws IOException,
         ShapeNotMatchException, InvalidRangeException {
      // testCopyVsCreate(root);
      testIterationOverRegion(root);
   }

   public static void testCopyVsCreate(ILogicalGroup root) throws IOException,
         ShapeNotMatchException, InvalidRangeException {
      System.out
            .println("------------------------------------------------------------------");
      System.out
            .println("------------------------------------------------------------------");
      System.out.println("Displaying images: ");
      IKey key = Factory.getFactory(root.getFactoryName())
            .createKey("images");
      ILogicalGroup data_grp = root.getGroup("data");
      List<IPathParameter> params = data_grp.getParameterValues(key);
      if (params.size() > 0) {
         key.pushParameter(params.get(0));
      }
      IDataItem data = data_grp.getDataItem(key);
      data = data_grp.getDataItem(key);
      if (data != null) {
         int[] shape = data.getShape();
         int[] origin = new int[data.getShape().length];
         IArray a = null;

         if (data != null && shape.length > 2) {
            try {
               // Parsing a region of the stack of images
               for (int i = 0; i < data.getRank() - 2; i++) {
                  shape[i] = 1;
                  origin[i] = 0;
               }

               a = data.getData(origin, shape);
               a = a.getArrayUtils().reduce().getArray();
               System.out.print("Displaying region [");
               for (int size : a.getShape()) {
                  System.out.print(size + ", ");
               }
               System.out.print("] at position [");
               for (int size : a.getIndex().getOrigin()) {
                  System.out.print(size + ", ");
               }
               System.out.print("]: ");

               StringBuffer buf = new StringBuffer();
               StringBuffer buf2 = new StringBuffer();
               Object oo = a.getArrayUtils().copyToNDJavaArray();
               IArray arr = Factory.getFactory(root.getFactoryName())
                     .createArray(a.getElementType(), a.getShape(), oo);
               IArrayIterator imageIterator1 = a.getIterator();
               IArrayIterator imageIterator2 = arr.getIterator();
               int i = 0;

               if (a.getElementType().equals(Integer.TYPE)) {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.getIntNext() + ",  ");
                     buf2.append(imageIterator2.getIntNext() + ",  ");
                  }
               } else if (a.getElementType().equals(Short.TYPE)) {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.getShortNext() + ",  ");
                     buf2.append(imageIterator2.getShortNext() + ",  ");
                  }
               } else if (a.getElementType().equals(Double.TYPE)) {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.getDoubleNext() + ",  ");
                     buf2.append(imageIterator2.getShortNext() + ",  ");
                  }
               } else if (a.getElementType().equals(Long.TYPE)) {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.getLongNext() + ",  ");
                     buf2.append(imageIterator2.getLongNext() + ",  ");
                  }
               } else if (a.getElementType().equals(Float.TYPE)) {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.getFloatNext() + ",  ");
                     buf2.append(imageIterator2.getFloatNext() + ",  ");
                  }
               } else {
                  while (imageIterator1.hasNext() && i++ < 1000) {
                     buf.append(imageIterator1.next() + ",  ");
                     buf2.append(imageIterator2.next() + ",  ");
                  }
               }

               int l1 = buf.toString().length() < 1000 ? buf.toString()
                     .length() : 1000;
               int l2 = buf2.toString().length() < 1000 ? buf2.toString()
                     .length() : 1000;
               System.out.println(buf.toString().substring(0, l1));
               System.out.println(buf2.toString().substring(0, l2));
               System.out
                     .println("Getting a copy of the region and create a new IArray with following shapes:");
               System.out.println("Region: " + a.getIndex());
               System.out.println("Copy: " + arr.getIndex());
               if (buf.toString().equals(buf2.toString())) {
                  System.out
                        .println("Both arrays have equivalent contents ");
               } else {
                  throw new IOException(
                        "Copying and creating a new array from a region of another array isn't equivalent!\nContents are different!!!");
               }
            } catch (IOException e) {
               e.printStackTrace();
            } catch (InvalidRangeException e) {
               e.printStackTrace();
            }
         }
      }
   }

   public static void testIterationOverRegion(ILogicalGroup root)
         throws IOException, InvalidRangeException {
      ILogicalGroup detectors = root.getGroup("detectors");
      ILogicalGroup detector;
      ILogicalGroup data_grp = root.getGroup("data");
      IDataItem data;
      IKey key;

      // Scanning "detectors" group
      System.out.println("Scanning \"detectors\" group");
      if (detectors != null) {
         for (IKey key1 : detectors.getDictionary().getAllKeys()) {
            detector = detectors.getGroup(key1);
            for (IKey key2 : detector.getDictionary().getAllKeys()) {
               data = detector.getDataItem(key2);
               if (data != null) {
                  System.out.println("------------------------------\n"
                        + data.toStringDebug());
               }
            }
         }
      }

      // Scanning "data" group
      System.out.println("Scanning \"data\" group");
      if (data_grp != null) {
         key = Factory.getFactory(root.getFactoryName()).createKey("images");
         data = data_grp.getDataItem(key);
         if (data != null) {
            int x = data.getShape()[data.getRank() - 2] / 2;
            int y = data.getShape()[data.getRank() - 1] / 2;
            int[] shape = new int[data.getRank()];
            int[] origin = new int[data.getRank()];
            shape[data.getRank() - 2] = x;
            shape[data.getRank() - 1] = y;
            for (int i = 0; i < data.getRank() - 2; i++) {
               shape[i] = 1;
            }

            IArray a = null;
            if (data != null) {
               System.out.println("------------------------------\n"
                     + data.toStringDebug());
               try {
                  a = data.getData(origin, shape);
                  System.out.println(a.getIndex());
                  IArrayIterator iter = a.getIterator();
                  for (int i = 0; i < x; i++) {
                     System.out.print("raw n°" + i + ":  ");
                     for (int j = 0; j < y; j++) {
                        System.out.print(iter.next() + ",  ");
                     }
                     System.out.println("");
                  }
               } catch (IOException e) {
                  e.printStackTrace();
               } catch (InvalidRangeException e) {
                  e.printStackTrace();
               }
            }
         }
      }
   }

}
