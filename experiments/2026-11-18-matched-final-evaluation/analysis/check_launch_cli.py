"""Exercise actual private runner parser -> actual wrapper parser, no CUDA."""
import contextlib,importlib.util,io,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import run_calibration as c
import admit_exec as a
runner_path=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('private_runner',runner_path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks=0
for backend in [
 ['prunets','index path','queries','timing.csv','top.tsv','500','10','0','0','1','0','256','256','-','64'],
 ['range','index','queries','timing.csv','top.tsv','500','10','0','1','1','0','256','256','gsp'],
 [sys.executable,'compact.py','--variant','monolithic','--index','index path','--verify','0'],
 ['bmp','--index','index path','--queries','queries','--cpu','0','--k','10']]:
    command=list(map(str,c.admission_command(Path('/source path'),Path('/log path'),backend)))
    def execute(ledger,experiment,argv,gate,outputs):
        assert argv==command and gate==600 and outputs==['/log path']
        with patch.object(sys,'argv',argv[1:]),patch.object(a,'run',return_value=0) as run:
            assert a.main()==0
            assert run.call_args.args==(str(Path('/log path')),backend)
        return {'payload':{'status':'EXECUTED'}}
    with patch.object(sys,'argv',['research.py','run','EXPERIMENT','--gate-seconds','600','--output','/log path','--command',*command]),patch.object(r,'Ledger'),patch.object(r,'execute',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
        r.main()
    checks+=1
print(json.dumps(dict(checks=checks,status='PASS',scope='Actual runner and wrapper parsers with generated controller argv; execution mocked, no GPU')))
