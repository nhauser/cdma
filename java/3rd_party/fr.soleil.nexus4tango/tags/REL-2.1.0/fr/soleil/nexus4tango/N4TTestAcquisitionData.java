package fr.soleil.nexus4tango;

import java.io.File;

import org.nexusformat.NexusException;

/**
 * Only a main procedure to run a sample
 *
 * @author rodriguez
 *
 */
public class N4TTestAcquisitionData {
	/**
	 * Main
	 *
	 * @param args
	 */
	public static String SAMPLES_DIR = "../../Samples/";
	public final static int BUFFSIZE = 100;

	public static void main(String[] args) {
		System.out.print("Hello starting work:");
        
        if( args.length > 0 && ! args[0].isEmpty() ) {
            SAMPLES_DIR = args[0];
        }
        System.out.println(SAMPLES_DIR);
		String file = SAMPLES_DIR;
		long time = System.currentTimeMillis();

		for( int iteration = 0; iteration < 1; iteration++)
		{
			try {
				// Initialization
				AcquisitionData n4t = new AcquisitionData(BUFFSIZE);
				AcquisitionData n4t2;
				DataItem ds;
				File fSamplesFolder = new File(file);
				File[] files = fSamplesFolder.listFiles();
				int iNumFile = 0 ;

				for( iNumFile = 0; iNumFile < files.length; iNumFile++)
				{
					file = files[iNumFile].getAbsolutePath();
					if( ! file.substring(file.length() - 4).equals(".nxs") )
						continue;

					// Starting process
					System.out.println("\n---------------------------------------------\n---------------------------------------------");
					System.out.println("************* Working on file: " + file);
					System.out.println("\n---------------------------------------------\n---------------------------------------------");
					n4t.setFile(file);
					
					PathNexus pgTargPath = new PathData(new String[] { "acqui_00001", "scan_data1", "turlututu" }, "turlututu");
					PathNexus pgDestPath = new PathData(new String[] { "acqui_00002"}, "link");
					
					n4t.writeLink(pgTargPath, pgDestPath);
					n4t.finalize();
					if( true )
						return;
					
					
					String[] list = n4t.getAcquiList(), list2;
					System.out.println("Number of acquisition: " + list.length);
					int image_count;
					System.out.println("Acquisition list:");
					for (int j = 0; j < list.length; j++)
					{
						if( j > 10 )
							System.out.println(j);
						System.out.println("\n---------------------------------------------\nScanning acquisition\n---------------------------------------------");
						System.out.println("-> Acquisition " + list[j] + " contains:");
						list2 = n4t.getInstrumentList(list[j]);
						image_count = n4t.getImageCount(list[j]);
						System.out.println("    - " + list2.length + " instruments");
						System.out.println("    - " + image_count + " images");
/*                    
						DataItem[] tab = new DataItem[10];
						for( int imm = 0; imm < 10; imm++ )
						{
							tab[imm] = n4t.getData2D(imm, list[j]);
						}
						Object o;
						for( int imm = 0; imm < 10; imm++ )
						{
							System.out.println(tab[imm]);
							o = tab[imm].getData();
						}
*/						
						System.out.println("\n---------------------------------------------\nDisplaying images data\n---------------------------------------------");
						n4t2 = new AcquisitionData(BUFFSIZE);
						n4t2.setFlagSingleRaw(false);
						n4t2.setFile(file);
						for (int im = 0; im < image_count; im++) {
							long lLcTime = System.currentTimeMillis();
							ds = n4t.getData2D(im, list[j]);
							System.out.println("Loading image at: "+n4t.getImagePath(list[j], im));
							System.out.println("Loaded in "+ ((System.currentTimeMillis() - lLcTime) / 1000.) +"s" );
							System.out.println(ds);
//							n4t2 = null;
							ds.finalize();
							ds = null;
						}
						n4t2 = null;
						System.out.println("---------------------------------------------\nDisplaying instruments infos\n---------------------------------------------");
						for (int h = 0; h < list2.length; h++)
						{
							if( h == 2 )
								System.out.println("here");
							System.out.println("* Getting instrument datas from: " + list2[h] );
							long lLcTime = System.currentTimeMillis();
							DataItem[] d = n4t.getInstrumentData(list[j], list2[h]);
							System.out.println("  - contains " + d.length + " items, loaded in " + ((System.currentTimeMillis() - lLcTime) / 1000.) + "s :");
							for (int l = 0; l < d.length; l++) {
								lLcTime = System.currentTimeMillis();
								if( d[l].getNodeName() != null && !d[l].getNodeName().equals("") )
								{
									n4t2 = new AcquisitionData(BUFFSIZE);
									n4t2.setFile(file);
									ds = n4t.getDataItem(list[j], list2[h], d[l].getNodeName(), true);
									n4t2 = null;
								}
								else
									ds = d[l];
								if( ds == null ) throw new NexusException("DataItem null !!!!");
								System.out.println("\n  Instrument's item number " + l + " loaded in "+ ((System.currentTimeMillis() - lLcTime) / 1000.) +"s\n" + ds);
								ds.finalize();
								ds = null;
							}
							d = null;
						}
					}
				}
				n4t.finalize();
			} catch (NexusException ne) {
				System.out.println("NexusException caught : Something was wrong!\n");
				ne.printStackTrace();
				return;
			} catch (Throwable e) {
				System.out.println("Throwable caught : Something was wrong!\n");
				e.printStackTrace();
				return;
			}
			System.out.println("\n\n**********************************************************\n"+iteration+"\n**********************************************************");
		}

		System.out.println("Work's done!");
		System.out.println("Total execution time: " + (System.currentTimeMillis() - time) / 1000. + "s");

		System.out.println("Bye bye ;-)");
	}
}
