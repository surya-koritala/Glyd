"""CI regressions for immutable inputs, exact rounded values and incomplete results."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from protocol import MODEL, REVISION, POLICY, TOKENS, check_inputs, complete_report, digest, plan

BINARY=Path(sys.argv.pop(1)).resolve()

class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/"fixtures").mkdir()
        raw=self.root/"source.bf16"
        words=list(range(65536))
        raw.write_bytes(struct.pack("<65536H",*words))
        self.case={"id":"edge","shape":[64,1024],"kind":"all","products":False,"exact_product":False}
        directory=self.root/"fixtures"/"edge"
        subprocess.run([str(BINARY),str(raw),"64","1024",str(directory)],check=True)
        self.case["files"]={p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in directory.iterdir()}
        self.manifest={"schema":1,"model":MODEL,"revision":REVISION,"tokens":TOKENS,"policy":POLICY,"cases":[self.case]}
        self.save()
    def tearDown(self):
        self.tmp.cleanup()
    def save(self):
        (self.root/"cases.json").write_text(json.dumps(self.manifest))
    def test_all_rounded_words(self):
        check_inputs(self.root)
        got=struct.unpack("<65536H",(self.root/"fixtures/edge/rounded.bf16").read_bytes())
        for word,rounded in enumerate(got):
            if word&0x7f80==0x7f80:
                expected=word
            else:
                expected=(word+7+((word>>4)&1))&0xfff0
                if expected&0x7fff>=0x7f80:
                    expected=(word&0x8000)|0x7f70
            self.assertEqual(rounded,expected)
    def test_corrupt_or_missing_bytes_refused(self):
        p=self.root/"fixtures/edge/compact.data"
        with p.open("r+b") as f:
            first=f.read(1);f.seek(0);f.write(bytes([first[0]^1]))
        with self.assertRaisesRegex(ValueError,"digest"):
            check_inputs(self.root)
        p.unlink()
        with self.assertRaises(ValueError):
            check_inputs(self.root)
    def test_policy_and_identity_refused(self):
        for field,value in (("revision","0"*40),("tokens",[1]),("policy",{})):
            old=self.manifest[field]; self.manifest[field]=value;self.save()
            with self.assertRaises(ValueError):
                check_inputs(self.root)
            self.manifest[field]=old
    def test_memory_accounting_refused(self):
        p=self.root/"fixtures/edge/pack.json"; meta=json.loads(p.read_text())
        meta["weight_buffer_bytes"]+=1;p.write_text(json.dumps(meta))
        self.case["files"]["pack.json"]={"bytes":p.stat().st_size,"sha256":digest(p)};self.save()
        with self.assertRaisesRegex(ValueError,"accounting"):
            check_inputs(self.root)
    def test_real_shape_plan(self):
        inventory=json.loads((Path(__file__).resolve().parents[1]/"lossy-2026/qwen38-inventory.json").read_text())
        cases,unsupported=plan(inventory)
        self.assertEqual(len(cases),15)
        self.assertTrue(unsupported)
        embedding=next(c for c in cases if "embed_tokens" in c["name"])
        self.assertEqual(embedding["shape"],[4096,5120])
        self.assertEqual(embedding["source_shape"],[248320,5120])
        self.assertFalse(embedding["whole_tensor"])
        self.assertTrue(any(c["shape"][1]%64 for c in cases))
    def test_incomplete_or_duplicate_results_refused(self):
        result={"case":"edge","complete":True,"unpack_bit_exact":True,"products":[]}
        self.assertTrue(complete_report(self.manifest,[result])["complete"])
        for rows in ([],[result,result],[{**result,"complete":False}],[{**result,"unpack_bit_exact":False}]):
            with self.assertRaises(ValueError):
                complete_report(self.manifest,rows)
    def test_expected_refusals_are_required(self):
        case={"id":"tail","shape":[192,80],"products":True,"exact_product":True}
        manifest={"cases":[case]}
        results=[]
        for m in TOKENS:
            for bias in (False,True):
                row={"tokens":m,"bias":bias,"status":"expected-refusal" if m>64 else "pass"}
                if m<=64:
                    row.update({"repeat_bit_exact":True,"reference_pass":True,
                                "integer_or_reference_bit_exact":True,"base_reference_bit_exact":True,
                                "reference_max":1.0,"max_abs_error":0.0,"normalized_rms_error":0.0,
                                "base_max_abs_error":0.0,"base_normalized_rms_error":0.0})
                    row.update({k:[0.1]*5 for k in ("compact_ms","expanded_ms","bf16_ms","unpack_ms","fp8_ms")})
                results.append(row)
        record={"case":"tail","complete":True,"unpack_bit_exact":True,"products":results}
        complete_report(manifest,[record])
        bad=copy.deepcopy(record);bad["products"].pop()
        with self.assertRaises(ValueError): complete_report(manifest,[bad])
        bad=copy.deepcopy(record);bad["products"][0]["normalized_rms_error"]=0.1
        with self.assertRaises(ValueError): complete_report(manifest,[bad])
        bad=copy.deepcopy(record);bad["products"][0]["integer_or_reference_bit_exact"]=False
        with self.assertRaises(ValueError): complete_report(manifest,[bad])
        bad=copy.deepcopy(record);del bad["products"][0]["fp8_ms"]
        with self.assertRaises(ValueError): complete_report(manifest,[bad])
        ok=copy.deepcopy(record);del ok["products"][0]["fp8_ms"];ok["products"][0]["fp8_error"]="unsupported"
        complete_report(manifest,[ok])
    def test_large_product_variants_are_required(self):
        case={"id":"wide","shape":[64,128],"products":True,"exact_product":False}
        timed={k:[0.1]*5 for k in ("compact_ms","expanded_ms","bf16_ms","unpack_ms","fp8_ms")}
        common={"status":"pass","repeat_bit_exact":True,"reference_pass":True,"reference_max":1.0,
                "max_abs_error":0.0,"normalized_rms_error":0.0,"base_max_abs_error":0.0,"base_normalized_rms_error":0.0}
        variant={"repeat_bit_exact":True,"reference_pass":True,"ms":[0.1]*5}
        rows=[{"tokens":m,"bias":b,**common,**timed,**({"variants":{"1":dict(variant),"2":dict(variant)}} if m>64 else {})}
              for m in TOKENS for b in (False,True)]
        record={"case":"wide","complete":True,"unpack_bit_exact":True,"products":rows}
        complete_report({"cases":[case]},[record])
        for change in (lambda r: r["variants"].pop("2"), lambda r: r["variants"]["1"].update(reference_pass=False),
                       lambda r: r["variants"]["2"].update(ms=[0.1]*4)):
            bad=copy.deepcopy(record);change(bad["products"][-1])
            with self.assertRaises(ValueError): complete_report({"cases":[case]},[bad])

if __name__=="__main__":
    unittest.main()
