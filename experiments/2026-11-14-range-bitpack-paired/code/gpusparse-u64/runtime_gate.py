"""Public Python handshake compatible with the project's registered runtime gate."""
import json,os,time
from pathlib import Path

def write_new(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def runtime_report(config,behavior):
 if not os.environ.get('RESEARCH_RUNTIME_DIR'):raise RuntimeError('Registered runtime directory is required')
 directory=Path(os.environ['RESEARCH_RUNTIME_DIR']);assert directory.is_dir()
 assert not (directory/'runtime.json').exists()
 temporary=directory/'runtime.pending.json';write_new(temporary,dict(config=config,behavior=behavior));temporary.rename(directory/'runtime.json')
 until=time.monotonic()+float(os.environ.get('RESEARCH_GATE_SECONDS','60'))
 while time.monotonic()<until:
  if (directory/'reject').exists():raise RuntimeError('Research runtime gate rejected')
  if (directory/'continue').exists():return
  time.sleep(.05)
 raise RuntimeError('Research runtime approval timed out')
