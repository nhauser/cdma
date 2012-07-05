package org.gumtree.data.test;

import java.io.File;
import java.io.IOException;
import java.util.Vector;

import org.gumtree.data.Factory;
import org.gumtree.data.interfaces.IDataset;

public class MainTest {
    protected static String plugin;
    protected static IDataset dataset;
    protected static String sample_path;
    private static Vector<Thread> poolThread = new Vector<Thread>();
    private static boolean useThreads = false;

    /**
     * Main process
     * 
     * @param args path to reach file we want to parse
     * @throws IOException
     * @throws FileAccessException
     *             args[0] = folder of application's dictionary view args[1] =
     *             plugin's name args[2] = folder where to find file
     */
    public static void main(String[] args) throws Exception {
        System.out.println("OK");

        final long time = System.currentTimeMillis();
        final String dico_path;
        int nbFiles = 0;
        plugin = "org.gumtree.data.soleil.NxsFactory";

        // Dictionary path
        if (args.length > 0) {
            dico_path = args[0];
            Factory.setDictionariesFolder(dico_path);
        }

        // Setting plug-in to use
        if (args.length > 1) {
            plugin = args[1];
        }

        if (args.length > 2) {
            sample_path = args[2];
        }
        int i = 0;
        File folder = new File(sample_path);
        if (folder.exists()) {
            for (File sample : folder.listFiles()) {

                Runnable test = new Process(sample);
                Thread thread = new Thread(test);
                thread.start();
                poolThread.add(thread);
                if( ! useThreads ) {
                    thread.join();
                }
                System.out.println(thread.getId() + "-------> " + sample.getName());

                if( true ) return;

                //test.run();
            }
        }

        if( useThreads ) {
            for( Thread t : poolThread ) {
                t.join();
            }
        }
        i++;
        // IFactory factory = Factory.getFactory(plugin);
        // TestArrayManual.testManuallyCreatedArrays(factory);
        System.out.println("Total execution time: "
                + ((System.currentTimeMillis() - time) / 1000.) + "s");
        System.out.println("Nb files tested: " + nbFiles);
    }
}
