"""Inventory integrity regressions; synthetic byte fixtures only, on CI."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from inventory import tensor_header, aggregate


class Inventory(unittest.TestCase):
    def test_shard_ranges_and_identity_are_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors'
            header={'weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}
            expected={'weight':{'dtype':'BF16','shape':[2]}}
            def write(h, data=b'\x00\x00\x80\x3f'):
                encoded=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
            write(header)
            rows=tensor_header(path,{'weight':path.name},expected)
            self.assertEqual(rows[0]['bytes'],4)
            with self.assertRaises(ValueError): tensor_header(path,{'weight':'different.safetensors'},expected)
            with self.assertRaises(ValueError): tensor_header(path,{'weight':path.name,'missing':path.name},expected)
            write(header,b'\x00')
            with self.assertRaises(ValueError): tensor_header(path,{'weight':path.name},expected)
            header['weight']['data_offsets']=[1,5];write(header,b'\x00'*5)
            with self.assertRaises(ValueError): tensor_header(path,{'weight':path.name},expected)

    def test_byte_accounting_refuses_omissions_and_unverified_data(self):
        row={'name':'weight','source_bytes':8,'bytes':8,'packed_bytes':48,'values':4,'changed_values':1,
             'tensor_header_bytes':28,'block_header_bytes':12,'dictionary_bytes':0,'field_bytes':0,
             'exception_bytes':0,'raw_bytes':8,'empirical_iid_entropy_bits':2,'exact_after_rounding':True}
        self.assertEqual(aggregate([row])['packed_bytes'],48)
        for bad in [[],[row,row],[{**row,'packed_bytes':49}],[{**row,'exact_after_rounding':False}]]:
            with self.assertRaises(ValueError): aggregate(bad)


if __name__ == '__main__':
    unittest.main()
