"""Host protocol checks, run on GitHub CI."""
import copy
import math
import unittest
from prepare_quality import choose_windows
from quality_compare import paired, control_passes


class Protocol(unittest.TestCase):
    def test_windows_are_held_out_disjoint_and_targets_follow_prefix(self):
        tokens = list(range(8000))
        rows = choose_windows(tokens)
        self.assertEqual(len({r['dataset_token_offset'] for r in rows}), 8)
        self.assertEqual(sum(len(r['input_ids'])-r['score_start'] for r in rows), 2048)
        self.assertEqual(rows[0]['input_ids'][rows[0]['score_start']], 256)
        for row in rows:
            start = row['dataset_token_offset']
            self.assertEqual(row['input_ids'], tokens[start:start+512])
        for a, b in zip(rows, rows[1:]):
            self.assertGreaterEqual(b['dataset_token_offset'], a['dataset_token_offset']+512)
        self.assertEqual(rows[-1]['input_ids'][-1], tokens[-1])
        with self.assertRaises(ValueError): choose_windows(list(range(4095)))
        with self.assertRaises(ValueError): choose_windows(tokens, prefix=512)
        with self.assertRaises(ValueError): choose_windows(tokens, count=0)
        expanded=choose_windows(list(range(50000)),count=32,length=1024,prefix=512)
        self.assertEqual(sum(len(r['input_ids'])-r['score_start'] for r in expanded),16384)
        for left,right in zip(expanded,expanded[1:]):
            self.assertGreaterEqual(right['dataset_token_offset']-left['dataset_token_offset'],1024)


    def test_pairing_and_nll_direction(self):
        a={'model':'Qwen/Qwen3.8-27B','revision':'pinned','evaluation_sha256':'same',
           'samples':[{'id':'a','score_lp':[-1.0,-2.0],'score_top1':[1,2]},
                      {'id':'b','score_lp':[-3.0],'score_top1':[3]}]}
        b=copy.deepcopy(a)
        self.assertTrue(control_passes(paired(a,b)))
        for r in b['samples']: r['score_lp']=[v-0.2 for v in r['score_lp']]
        result=paired(a,b)
        self.assertAlmostEqual(result['mean_nll_delta'],0.2)
        self.assertAlmostEqual(result['perplexity_ratio'],math.exp(0.2))
        self.assertFalse(control_passes(result))
        self.assertEqual(result['quality_evidence'],'degradation_detected')
        self.assertEqual(paired(a,a)['quality_evidence'],'no_worse_on_this_screen')
        b['evaluation_sha256']='changed'
        with self.assertRaises(ValueError): paired(a,b)
        b=copy.deepcopy(a);b['samples'][0]['score_lp'].pop()
        with self.assertRaises(ValueError): paired(a,b)
        b=copy.deepcopy(a);b['samples'][0]['score_lp'][0]=float('nan')
        with self.assertRaises(ValueError): paired(a,b)


if __name__ == '__main__':
    unittest.main()
