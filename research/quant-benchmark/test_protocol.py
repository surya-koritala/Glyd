"""Protocol regressions; run on CI, never on a workstation."""
import copy
import unittest
from protocol import choice_metrics, decision, digest, paired_interval, validate_arm, validate_question


def question():
    row={'id':'science/0','group':'science','task':'science','metric':'acc_norm','gold':[1],
         'choice_lengths':[1,10],'requests':[{'context_ids':[1,2],'continuation_ids':[3]},
                                           {'context_ids':[1,2],'continuation_ids':[4,5]}]}
    row['sha256']=digest(row)
    return row


class ProtocolTests(unittest.TestCase):
    def test_official_length_normalization_and_tie_order(self):
        q=question()
        self.assertEqual(choice_metrics(q,[-1.,-2.]),{'acc':0,'acc_norm':1,'prediction':0,'prediction_norm':1})
        self.assertEqual(choice_metrics(q,[-1.,-10.])['prediction_norm'],0)
        with self.assertRaises(ValueError): choice_metrics(q,[float('nan'),-1.])
        with self.assertRaises(ValueError): choice_metrics(q,[-1.])

    def test_input_mutation_and_truncation_are_refused(self):
        q=question();validate_question(q,3)
        with self.assertRaises(ValueError): validate_question(q,2)
        q['gold']=[0]
        with self.assertRaises(ValueError): validate_question(q,3)

    def test_identical_observations_do_not_claim_zero_uncertainty(self):
        value=paired_interval([1]*1000,[1]*1000,.01)
        self.assertEqual(value['accuracy_loss'],0)
        self.assertLess(value['loss_interval'][0],0)
        self.assertGreater(value['loss_interval'][1],0)
        self.assertLess(value['loss_interval'][1],.01)

    def test_paired_losses_preserve_sign_and_pairing(self):
        value=paired_interval([1]*100+[0]*20,[0]*100+[1]*20,.05)
        self.assertEqual(value['harm_count'],100)
        self.assertEqual(value['gain_count'],20)
        self.assertGreater(value['loss_interval'][0],0)
        with self.assertRaises(ValueError): paired_interval([1],[1,0],.05)

    def test_weak_task_cannot_hide_in_a_good_average(self):
        policy={'max_task_loss_pp':1.,'max_macro_loss_pp':.5}
        c={'bf16':{'groups':{'science':{'loss_interval':[.02,.04]}},'macro_loss_interval':[-.1,-.05]}}
        self.assertEqual(decision(c,policy)['quality_decision'],'refocus')
        c['bf16']['groups']['science']['loss_interval']=[-.001,.02]
        self.assertEqual(decision(c,policy)['quality_decision'],'inconclusive')
        c['bf16']['groups']['science']['loss_interval']=[-.001,.001]
        self.assertEqual(decision(c,policy)['quality_decision'],'continue')
        self.assertEqual(decision(c,policy)['overall_decision'],'incomplete')

    def test_incomplete_or_mixed_arm_cannot_establish_baseline(self):
        q=question()
        index={k:q[k] for k in ('id','sha256','task','group','metric','gold','choice_lengths')}|{'choices':2}
        m={'schema':1,'profile':'screen','suite_sha256':'a','model_key':'qwen38-27b','model_sha256':'b',
           'request_manifest_sha256':'c','source_commit':'d','question_count':1,'question_index':[index]}
        h={k:m[k] for k in ('schema','profile','suite_sha256','model_key','model_sha256','request_manifest_sha256','source_commit')}
        h.update(arm='bf16',runtime='common-bf16-eager',complete=True,errors=0)
        row={k:q[k] for k in ('id','sha256','task','group','metric')}
        row.update(scores=[-1.,-2.],correct=1,**choice_metrics(q,[-1.,-2.]))
        validate_arm(m,h,[row])
        for key,value in [('complete',False),('model_sha256','changed'),('source_commit','changed'),('runtime','native-fp8')]:
            altered=copy.deepcopy(h);altered[key]=value
            with self.assertRaises(ValueError): validate_arm(m,altered,[row])
        with self.assertRaises(ValueError): validate_arm(m,h,[row,row])
        altered=copy.deepcopy(row);altered['correct']=0
        with self.assertRaises(ValueError): validate_arm(m,h,[altered])


if __name__=='__main__': unittest.main()
