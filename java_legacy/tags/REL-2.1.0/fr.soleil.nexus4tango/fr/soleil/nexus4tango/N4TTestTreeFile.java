package fr.soleil.nexus4tango;

import org.nexusformat.*;

import java.io.File;
import java.util.Random;


/**
 * Only a main procedure to run a sample
 *
 * @author rodriguez
 *
 */

public class N4TTestTreeFile {
	private static final int HUDGE_TAB_SIZE = 2048; // 8192
	private static final int BIG_TAB_SIZE 	= 1024; // 4096
	private static final String TEST_FILE 	= "../../Samples/Test/TestTreeFile.nxs";
	private static final String LINKED_FILE = "../../Samples/SWING_Samples/Louis_16_2009-07-15_15-49-24.nxs";
	//private static final String LINKED_FILE = "../../Samples/Testard_2007-12-16_18-11-25.nxs";
	//private static final String LINKED_FILE = "../../Samples/Tortorici_50_2009-09-26_22-13-06.nxs";

	/**
	 * Main
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("Hello starting work:");
		// Some clean up
		String file;
		long time = System.currentTimeMillis();
		file = TEST_FILE;
		File dest_file = new File(file);

		if (dest_file.exists() && !dest_file.delete())
			System.out.println("Unable to delete file: " + file);

		dest_file = null;
		NexusFileBrowser.ShowFreeMem("No works done yet...");

		try {
			// Initialization
			DataItem dsData = null, dsData2 = null, dsData3 = null;
			PathData p1 = null, p2 = null, pData = null;
			PathGroup pg2 = null;

			// Reading data from a First file
			System.out.println("----------- Reading -----------");
			AcquisitionData tf = new AcquisitionData(LINKED_FILE);
			tf.setCompressedData(true);
			System.out.println("Reading datas from: " + LINKED_FILE );
			dsData  = tf.getData2D(1, "/D1A_016_D1A");

			System.out.println(tf.getImagePath("D1A_016_D1A", 2));
			dsData2 = tf.getData2D(2, "/D1A_016_D1A");
			dsData3 = tf.getData2D(3, "/D1A_016_D1A");
			System.out.println(dsData + "\n> Datas read... OK");

			System.out.println("Getting a path from a NXtechnical_data: ");
			dsData = tf.getDataItem("D1A_016_D1A", "ans__ca__machinestatus__#1", "lifetime", false);
			System.out.println("> Path read... OK   " + dsData.getPath());
			System.out.println("Reading the data directly from this path: ");
			dsData = tf.readData(dsData.getPath());
			System.out.println("> Data read... OK   " + dsData);

			// Setting another file as current
			System.out.println("Setting a new file for writing: "+ file);
			tf.setFile(file);

			String s = "tartampion est un nom souvent utilisé";
			PathData pdPath = new PathData(new String[]{"titi", "toto"}, "data");
			tf.writeData(s, pdPath);
			
			System.out.println("> New file OK");

			// Writing some data
			System.out.println("\n----------- Writing -----------");

/*
			pg1 = new PathGroup( new String[] {"D1A_016_D1A"} );
			pg2 = new PathGroup( new String[] {"system"} );
			System.out.println("Writing a GROUP link on an external acquisition...");
			tf.writeGroupLink(LINKED_FILE, pg1, pg2);

			System.out.println("> Link wrote... OK (case insensitivity OK)");
			if( true ) return;
			PathData pga1 = new PathData( new String[] {"system", "image#2"}, "data" );
			PathData pga2 = new PathData( new String[] {"system"}, "my_data" );
			System.out.println("Writing a GROUP link on an external acquisition...");
			
			tf.writeRelativeLink(pga1, pga2);

			// Copying the whole acquisition from LINKED_FILE into TEST_FILE
			System.out.println("Writing a DEEP COPY from an acquisition...");
			pg1.setFile(LINKED_FILE);
			pg2.pushNode(new NexusNode("deep_copy", "NXdata", true));
			tf.writeDeepCopy(pg1, pg2);
			System.out.println("> Deep copy... OK");

			tf.setFlagLinkCheck(true);

			System.out.println("Writing a DATA link on an external instrument...");
			p1 = new PathData("D1A_016_D1A", "avIex", "xBIN");
			p2 = new PathData(new String[] {"system"}, "technical_data_lnk");
			System.out.println("> Link created yet: " + tf.existPath(p2) );
			tf.writeDataLink(LINKED_FILE, p1, p2);
			System.out.println("> Link created yet: " + tf.existPath(p2) );
			System.out.println(tf.readData(p2));
			System.out.println("> Datalink OK");

			System.out.println("Writing multiple rebond data link (a data link on an existing data link)...");
			p1 = new PathData("D1A_016_D1A", "avIex", "xBIN");
			p2 = new PathData(new String[] {"system"}, "image_lnk");
			tf.writeDataLink(LINKED_FILE, p1, p2);

			tf.setFlagLinkCheck(false);

			p1 = new PathData(new String[] {"system"}, "image_lnk");
			p2 = new PathData(new String[] {"system", "titi", "toto"}, "image_lnk1");
			tf.writeDataLink(TEST_FILE, p1, p2);
			System.out.println("> Link wrote OK");
			d = tf.readData(p2);
			System.out.println(d);

			p1 = new PathData(new String[] {PathNexus.PARENT_NODE, PathNexus.PARENT_NODE, PathNexus.PARENT_NODE}, "image_lnk");
			p2 = new PathData(new String[] {"system", "titi", "toto"}, "image_lnk2");
			tf.writeDataLink(TEST_FILE, p1, p2);
			System.out.println("> Link wrote OK");
			d = tf.readData(p2);
			System.out.println(d);

			pr = new PathRelative(new String[] {PathNexus.PARENT_NODE, PathNexus.PARENT_NODE, PathNexus.PARENT_NODE}, "image_lnk");
			pn 	= new PathData(new String[] {"system", "titi", "toto"}, "image_lnk3");
			tf.writeRelativeLink(pr, pn);
			System.out.println("> Link wrote OK");
			d = tf.readData(pn);
			System.out.println(d);

			tf.setFlagLinkCheck(true);

			PathRelative pr1 = new PathRelative( new String[] {PathNexus.PARENT_NODE, PathNexus.PARENT_NODE, "<NXinstrument>"} );
			pg2 = new PathGroup( new String[] {"system", "group_link"} );
			System.out.println("Writing a GROUP link on an external instrument... using relative link");
			tf.writeRelativeLink(pr1, pg2);

			PathData pd = new PathData(new String[] {"system"}, "image_lnk");
			pn = new PathData(new String[] {"system", "titi", "toto"}, "image_lnk4");
			tf.writeRelativeLink(pd, pn);
			System.out.println("> Link wrote OK");
			d = tf.readData(pn);
			System.out.println(d);
*/
			int size1 = 2, size2 = 5, size3 = 8;
			double va = 0;
			double[] tab1 = new double[size1 * size2 * size3];
			double[][][] tab2 = new double[size1][size2][size3];
			
			for( int i = 0; i < size1 * size2 * size3; i++ ) {
				tab1[i] = va++;
			}
			va = 0;
			for( int i = 0; i < size1; i++ ) {
				for( int j = 0; j < size2; j++ ) { 
					for( int k = 0; k < size3; k++ ) {
						tab2[i][j][k] = va++;
					}
				}
			}
			tf.writeData(
					tab1,
					new PathData("system", "array", "mono_dim")
					);
			tf.writeData(
					tab2,
					new PathData("system", "array", "multi_dim")
					);
			
	
			NexusFileReader nfr = new NexusFileReader();
			tab2 = new double[size1][size2][size3];
			nfr.reshapeArray(new int[] {0, 0, 0}, new int[] {size1, size2, size3}, tab1, tab2);

			tf.writeData(
					tab2,
					new PathData("system", "array", "multi_dim_from_mono")
				);
			
			System.out.println("Writing data of different primitive type: ");
			short[][] tab  = new short[BIG_TAB_SIZE][BIG_TAB_SIZE];
			for( int iter = 0; iter < 5; iter++)
			{
				for(int o = 0; o < BIG_TAB_SIZE; o++)
					java.util.Arrays.fill(tab[o], (short) (new Random().nextInt() % 100));
				tf.writeData(
						tab,
						new PathData("system", "array", "big_array_" +iter)
						);
				tf.writeAttr(
						"name",
						"big enough array n�" + iter, new PathData("system", "array", "big_array_" +iter)
						) ;
				tf.writeAttr("numeraire1", 1234, new PathData("system", "array", "big_array_" +iter));
				System.out.println("> Wrote "+BIG_TAB_SIZE+"x"+BIG_TAB_SIZE+" image number " + iter + " OK");
			}

			tab = new short[HUDGE_TAB_SIZE][HUDGE_TAB_SIZE];
			for(int o = 0; o < HUDGE_TAB_SIZE; o++)
				java.util.Arrays.fill(tab[o], (short) (new Random().nextInt() % 100));
			tf.writeData(tab, new PathData("system", "array", "hudge_array"));
			System.out.println("> Wrote "+HUDGE_TAB_SIZE+"x"+HUDGE_TAB_SIZE+" OK");

			tf.writeData(null, new PathData("system", "values", "null"));
			System.out.println("> Wrote null value OK");

			if( dsData != null )
			{
				pData = new PathData("system", "image", "data1");
				tf.writeData(dsData, pData);
			}

			if( dsData2 != null )
			{
				pData = new PathData("system", "image", "data2");
				tf.writeData(dsData2, pData);
			}

			if( dsData3 != null )
			{
				pData = new PathData("system", "image", "data3");
				tf.writeData(dsData3, pData);
			}
			System.out.println("> Wrote DataItems read from another file OK");

			pData = new PathData("system", "primitive_single", "boolean");
			tf.writeData(true, pData);
			System.out.println("> Wrote a single boolean value");

			pData = new PathData("system", "primitive_single", "integer");
			tf.writeData(8, pData);
			System.out.println("> Wrote single integer OK");

			pData = new PathData("system", "primitive_single", "string");
			tf.writeData("Test of a single string", pData);
			System.out.println("> Wrote single string OK");

			boolean[][] b1 = {{false, false, false},{true, true, true}};
			boolean[][] b2 = {{true, false, true},{true, false, false}};
			boolean[][][] b3= {b1, b2};
			pData = new PathData("system", "array", "boolean");
			tf.writeData(b3, pData);
			System.out.println("> Wrote a 3 dim boolean array OK");

			pData = new PathData("system", "array", "double");
			tf.writeData(new double[] { 12756.2, 40075.02, 5974200000000000000000000. }, pData);
			System.out.println("> Wrote a 1 dim array of double OK");

			dsData = new DataItem( pData );
			pData = new PathData("system", "array", "link_on_double");
			pData.setFile(tf.getFile());
			tf.writeData(dsData, pData);
			System.out.println("> Wrote a link using the writeData method");
			
			// Reading some data
			System.out.println("----------- Reading -----------");
			System.out.println("Reading external link from file: " + file);

			p1 = new PathData(new String[] {"system", "titi", "toto"}, "image_lnk2");
			dsData = tf.readData( (PathData) p1 );
			System.out.println("> Read a multiple rebond (2 rebond) data link: " + dsData);

			String[] list;
			System.out.println("Scanning instrument from another file through a group link (on acquisition's level)...");
			list = tf.getInstrumentList("system");
			System.out.println("> Scan OK");
			System.out.println("Number of instrument found: " + list.length);
			for (int h = 0; h < list.length; h++) {
				System.out.println("   "+ h + "      " + list[h]);
			}
/*
	 		PathData pData1 = new PathData("system", "instrument", "titi_toto");
			PathRelative pRel = new PathRelative( new String[] {"..", "..", "array" });
			tf.writeRelativeLink(pRel, pData1);

			NexusFileBrowser nfb = new NexusFileBrowser(file);
			nfb.openFile();
			pData1.popNode();
			NexusNode[] nn = nfb.listChildren(new PathGroup("system"));
			System.out.println("Item found while listing the acquisition children:");
			for(int i=0; i < nn.length; i++)
				System.out.println("   " + i + " => " + nn[i].getNodeName());
			nfb.closeFile();
*/
			System.out.println("Reading an instrument through link:");
			dsData = tf.readData( (PathData) p2 );
			System.out.println("> Instrument read OK");
			System.out.println("Instrument contains:\n" + dsData);

			DataItem[] aData = tf.getInstrumentData("system", "aviex");
			System.out.println("Reading instrument Aviex datas from an the acquisition:");
			for (int h = 0; h < aData.length; h++) {
				System.out.println("  * Aviex item " + h + " contains: " + aData[h]);
			}

			System.out.println("Reading all descendant nodes of a group link:");
			pg2 = new PathGroup( new String[] {"system", "group_link"} );
			DataItem[] d1 = tf.readData(pg2);
			System.out.println("> Number of nodes read: " + d1.length);
			for(int j = 0; j < d1.length; j++ )
			{
				System.out.println(j + " >>> " + d1[j]);
			}

			String[] list2 = tf.getInstrumentList("system");
			System.out.println("Number of listed instrument with NO SELECTOR ON TYPE: " + list2.length);
			for (int h = 0; h < list2.length; h++) {
				System.out.println("  * Instrument item " + h + " from: " + list2[h]);
			}

			list2 = tf.getInstrumentList("system", AcquisitionData.Instrument.INTENSITY_MONITOR);
			System.out.println("\nNumber of listed instrument with INTENSITY_MONITOR FILTER ON TYPE: " + list2.length);
			for (int h = 0; h < list2.length; h++) {
				System.out.println("  * INTENSITY_MONITOR  Instrument item " + h + " from: " + list2[h]);
			}

			pData = new PathData("system", "values", "null");
			dsData = tf.readData(pData);
			System.out.println(dsData);
			System.out.println("> Read null value OK");

			pData = new PathData("system", "primitive_single", "string");
			dsData = tf.readData(pData);
			System.out.println("> Read single primitive string: " + dsData.getData() + " OK");

			pData = new PathData("system", "primitive_single", "integer");
			dsData = tf.readData(pData);
			System.out.println("> Read single primitive integer : " + dsData.getData() + " OK");

			pData = new PathData("system", "primitive_single", "boolean");
			dsData = tf.readData(pData);
			System.out.println("> Read single primitive boolean: " + dsData.getData() + " OK");

			pData = new PathData("system", "array", "boolean");
			dsData = tf.readData(pData);
			System.out.println("> Read boolean array OK \n" + dsData );

			pData = new PathData("system", "array", "double");
			dsData = tf.readData(pData);
			System.out.println("> Read array double: {" + ((double[]) dsData.getData())[0] + ", " + ((double[]) dsData.getData())[1] + ", " + ((double[]) dsData.getData())[2] + "}  OK");

			DataItem dsData4;
			for( int iter = 0; iter < 5; iter++)
			{
				pData = new PathData("system", "array", "big_array_" + iter);
				dsData4 = tf.readData(pData);
				System.out.println("> Read attribute name of the big array: " + tf.readAttr("name", pData) + " OK");
				System.out.println("> Read attribute numeraire1 of the big array: " + tf.readAttr("numeraire1", pData) + " OK");

				System.out.println("> Read big array n°" +iter+ " OK\n" + dsData4);
				dsData4.finalize();
				dsData4 = null;
			}

			pData = new PathData("system", "array", "hudge_array");
			dsData = tf.readData(pData);
			System.out.println("> Reading hudge array OK\n"+dsData);
			System.out.println(dsData);
			dsData = null;

			pData = new PathData("system", "image", "data1");
			if( tf.existPath(pData) )
				dsData = tf.readData(pData);
			System.out.println("> Reading image 1 copied from another file OK\n" + dsData);
			dsData = null;

			pData = new PathData("system", "image", "data2");
			if( tf.existPath(pData) )
				dsData = tf.readData(pData);
			System.out.println("> Reading image 2 copied from another file OK\n" + dsData);
			dsData = null;

			pData = new PathData("system", "image", "data3");
			if( tf.existPath(pData) )
				dsData = tf.readData(pData);
			System.out.println("> Reading image 3 copied from another file OK\n" + dsData);
			dsData = null;

			try { tf.finalize(); } catch(Throwable t) {t.printStackTrace();}
		} catch (NexusException ne) {
			System.out.println("NexusException caught : Something was wrong!\n");
			ne.printStackTrace();
			return;
		}catch (Exception e) {
			e.printStackTrace();
			return;
		}

		System.out.println("Work's done!");
		System.out.println("Elapsed time: "	+ (System.currentTimeMillis() - time) + "ms");
		System.out.println("Bye bye ;-)");
	}

}
