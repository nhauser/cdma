#unit test module for dataset

import unittest

import cdma

class DatasetTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_open(self):
        ds = cdma.open_dataset("file:../data/demo.nxs")
        print ds.location


