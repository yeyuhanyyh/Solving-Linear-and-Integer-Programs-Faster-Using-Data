"""Nonsolver tests: atomic artifacts, mocked interruption, training-only policy."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import run_followup as run


class RunnerTests(unittest.TestCase):
    def test_exclusive_atomic_publication(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT,prefix=".mock-test-") as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(run.ROOT))
            path=Path(tmp)/"marker.json"
            def interrupted(stream):
                stream.write(b'{"partial":');raise RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):run.publish_new(path,interrupted)
            self.assertFalse(path.exists())
            self.assertTrue(list(Path(tmp).glob("*.writing-*")))
            run.write_new(path,{"complete":True})
            old=path.read_bytes()
            with self.assertRaises(FileExistsError):run.write_new(path,{"complete":False})
            self.assertEqual(path.read_bytes(),old)

    def test_checkpoint_resume_and_hash_rejection(self):
        protocol=run.read(run.ROOT/"protocol.json");scenario=protocol["scenarios"][0]
        sources={"mock":"no real optimization"}
        template=run.physical_template(401,"mock",sources,scenario)
        witness=run.all_on_feasible_solution(template.original_base)
        counter={"calls":0}
        def fake_solve(*args,**kwargs):
            counter["calls"]+=1
            if counter["calls"]==17:raise RuntimeError("simulated interrupted seventeenth call")
            return SimpleNamespace(x=witness.copy(),certified=True,
                to_dict=lambda **kw:dict(certified=True,valid=True,status="TEST MOCK ONLY"))
        with tempfile.TemporaryDirectory(dir=run.ROOT,prefix=".mock-test-") as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(run.ROOT))
            out=Path(tmp)
            with patch.object(run.adapter,"solve",fake_solve):
                with self.assertRaises(RuntimeError):run.label_group(out,protocol,401,scenario,sources,run.Deadline(600))
            markers=list(out.glob("training/*/checkpoints/*/checkpoint-complete.json"))
            self.assertEqual(len(markers),1);self.assertEqual(run.read(markers[0])["training_count"],16)
            counter["calls"]=0
            def completed_mock(*args,**kwargs):
                counter["calls"]+=1
                return SimpleNamespace(x=witness.copy(),certified=True,
                    to_dict=lambda **kw:dict(certified=True,valid=True,status="TEST MOCK ONLY"))
            with patch.object(run.adapter,"solve",completed_mock):
                result=run.label_group(out,protocol,401,scenario,sources,run.Deadline(600))
            self.assertEqual(counter["calls"],80)
            self.assertEqual([p["training_count"] for p in result["prefixes"]],[16,32,48,64,80,96])
            folder=next((out/"training").iterdir())
            with (folder/"training.npz").open("ab") as stream:stream.write(b"tampered")
            with patch.object(run.adapter,"solve",side_effect=AssertionError("must reject before any solve")):
                with self.assertRaises(RuntimeError):run.label_group(out,protocol,401,scenario,sources,run.Deadline(600))

    def test_training_only_scenario_and_main_n(self):
        protocol=run.read(run.ROOT/"protocol.json");groups=[]
        for scenario in protocol["scenarios"]:
            for seed in protocol["physical_seeds"]:
                eligible=seed!=607 and scenario==protocol["scenarios"][0]
                groups.append(dict(group_id=f"{scenario['scenario_id']}-{seed}",scenario_id=scenario["scenario_id"],
                    physical_seed=seed,observed_covered_candidate=eligible,
                    prefixes=[dict(training_count=n,exact_K=20 if n>=64 else 19) for n in (16,32,48,64,96)],
                    all96_label_wall_seconds=1.,all96_label_cpu_seconds=.5))
        with tempfile.TemporaryDirectory(dir=run.ROOT,prefix=".mock-test-") as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(run.ROOT))
            result=run.freeze_training_screen(Path(tmp),protocol,groups)
            self.assertTrue(result["any_candidate"])
            self.assertEqual(result["scenarios"][0]["main_training_count"],64)
            self.assertEqual(result["scenarios"][0]["eligible_physical_seeds"],[401,503])
            self.assertFalse(result["test_information_used"])
            self.assertEqual(run.freeze_training_screen(Path(tmp),protocol,groups),result)
            self.assertTrue(all(not s["candidate"] for s in result["scenarios"][1:]))


if __name__=="__main__":unittest.main()
