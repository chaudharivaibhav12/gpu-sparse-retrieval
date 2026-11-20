import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import admit_exec as m
base=dict(contexts=[],free_mib=10000,available_kib=34*1024*1024,busy_cores=4)
assert m.eligible(base)
for key,value in [('contexts',[(123,100)]),('free_mib',9999),('available_kib',34*1024*1024-1),('busy_cores',4.01)]:
    assert not m.eligible(dict(base,**{key:value}))
class ExecReached(Exception):pass
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'admit.jsonl'
    with patch.object(m,'sample',return_value=base),patch.object(m.os,'execvpe',side_effect=ExecReached) as execute:
        try:m.run(path,['backend','a b'])
        except ExecReached:pass
        else:raise AssertionError('backend not executed')
    assert execute.call_args.args[:2]==('backend',['backend','a b'])
    assert execute.call_args.args[2]==dict(m.os.environ)
    assert json.loads(path.read_text())['decision']=='exec'
    with patch.object(m,'sample',return_value=dict(base,contexts=[(123,100)])),patch.object(m.time,'monotonic',side_effect=[0,121,121]),patch.object(m.os,'execvpe') as execute:
        assert m.run(Path(d)/'busy.jsonl',['backend'])==75
        execute.assert_not_called()
    with patch.object(m,'sample',return_value=base),patch.object(m.time,'monotonic',side_effect=[0,121,121]),patch.object(m.os,'execvpe') as execute:
        assert m.run(Path(d)/'expired.jsonl',['backend'])==75
        execute.assert_not_called()
    with patch.object(m,'sample',side_effect=AssertionError('wrong UUID')),patch.object(m.os,'execvpe') as execute:
        try:m.run(Path(d)/'invalid.jsonl',['backend'])
        except AssertionError:pass
        else:raise AssertionError('invalid GPU admitted')
        execute.assert_not_called()
print(json.dumps(dict(checks=9,result='PASS',qualification='Mock resource decisions/exec; no GPU access')))
