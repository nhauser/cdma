package org.gumtree.data.test;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.locks.ReentrantLock;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.exception.InvalidRangeException;
import org.gumtree.data.exception.ShapeNotMatchException;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.gumtree.data.util.performance.Benchmarker;

public class Process implements Runnable {
   
    private File mSample;
    private boolean mUseThread = true;
    
    private static ReentrantLock g_mutex; // Mutex for thread safety
    public Process( File sample ) {
       mSample = sample;
       if( g_mutex == null )
          g_mutex = new ReentrantLock();
    }
 
    @Override
    public void run() {

       IFactory factory = null;

      IDataset dataset = null;
      try {
         // Auto detecting factory
         factory = Factory.getFactory(mSample.toURI());

         if (factory == null) {
            System.out.println("Failed auto-detecting plugin: "   + mSample.getAbsolutePath());
            return;
         }

         // Create dataset according URI
         dataset = factory.createDatasetInstance(mSample.toURI());

      } catch (IOException ioe) {
         ioe.printStackTrace();
         return;
      } catch (Exception e) {
         e.printStackTrace();
         return;
      }

      IDataset DATASET = dataset;
      String pluginName = factory.getName();

      // Get file roots
      ILogicalGroup root = null;
      IGroup phy_root = null;

      // Execute tests
      // Displaying physical structure of the datasource
      setActiveView(DATASET, "NONE: physical parsing!!", pluginName);
      phy_root = DATASET.getRootGroup();
      TestStructure.displayStructure(phy_root);

      // Displaying 2 kind of view for data
      setActiveView(DATASET, "FLAT", pluginName);
      root = DATASET.getLogicalRoot();
      TestDictionaries.testKeys(root);

      setActiveView(DATASET, "HIERARCHICAL", pluginName);
      root = DATASET.getLogicalRoot();
      TestDictionaries.testKeys(root);

      // Displaying values associated to keys
      TestDictionaries.testValues(root);

      // Testing CubeData
      IDataItem item = root.getDataItem("images");
      if (item != null) {
         try {
            Runnable cube = new Cube(item);
            Thread thread = new Thread(cube);
            thread.start();
            
            if( ! mUseThread ) {
               thread.join();
            }
         } catch (Exception e) {
            e.printStackTrace();
         }
      }

      // Testing some datas
      try {
         TestData.testData(root);
      } catch (IOException e2) {
         e2.printStackTrace();
      } catch (ShapeNotMatchException e2) {
         e2.printStackTrace();
      } catch (InvalidRangeException e2) {
         e2.printStackTrace();
      }

      // Testing arrays iteration, copy, slice...
      try {
         TestArray.TestArray(pluginName, mSample.getAbsolutePath());
      } catch (Exception e1) {
         e1.printStackTrace();
      }

      System.out.println(Benchmarker.print());
    }

   public static void setActiveView(IDataset dset, String expName, String pluginName) {
      Factory.setActiveView(expName);
      System.out.println("------------------------------------------------------------------");
      System.out.println("----------------     " + expName );
      System.out.println("------------------------------------------------------------------");
      System.out.println("Active plugin: " + pluginName);
      System.out.println("Active view: " + Factory.getActiveView());
      System.out.println("View dict: " + Factory.getKeyDictionaryPath());
      System.out.println("Dict. folder: "   + Factory.getMappingDictionaryFolder(Factory.getFactory(pluginName)));
      System.out.println("Working on file: " + dset.getLocation());
      if( dset instanceof NxsDataset ) {
         System.out.println( "Using configuration mode: " + ((NxsDataset) dset).getConfiguration().getLabel() );
      }
   }
   
   private class Cube implements Runnable {
      private IDataItem m_item;
      
      public Cube( IDataItem item ) {
         m_item = item;
      }
      @Override
      public void run() {
         try {
            TestCubeData.testCube(m_item);
         } catch (Exception e) {
            e.printStackTrace();
         }
      }
   }
}
