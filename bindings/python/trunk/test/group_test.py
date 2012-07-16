#unit test for group wrappers

import unittest

import cdma


class group_test(unittest.TestCase):
    def setUp(self):
        self.ds = cdma.open_dataset("file:../data/demo.nxs")
        self.rg = self.ds.root_group

    def tearDown(self):
        pass

    def test_open(self):

        #fetching the root group
        rg = self.ds.root_group 
        print rg

    def test_iteration(self):
        
        g = self.rg["D1A_016_D1A"]
        self.assertTrue(g.short_name == "D1A_016_D1A")
        self.assertTrue(g.parent.short_name == "")
        self.assertTrue(g.root.short_name == "")

        #iterate over data items
        for d in cdma.get_dataitems(g):
            self.assertTrue(isinstance(d,cdma.DataItem))

        for g in cdma.get_groups(g):
            self.assertTrue(isinstance(g,cdma.Group))

        for d in cdma.get_dimensions(g):
            self.assertTrue(isinstance(d,cdma.Dimension))


        


        



