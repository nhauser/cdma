#test for data items

import unittest

import cdma

class dataitem_test(unittest.TestCase):
    def setUp(self):
        self.ds = cdma.open_dataset("file:../data/demo.nxs")
        self.rg = self.ds.root_group
        self.g  = self.rg["D1A_016_D1A"]

    def tearDown(self):
        pass

    def test_open(self):
        
        for d in self.g.items:
            print d
