"""Materialize the score-width ablation from byte-verified parent build sources.

No index conversion, dependency update, retrieval or benchmark execution.
"""
import argparse,hashlib,json,re,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ACCESSORS = """    pub(crate) fn project_ids(&self) -> &[u64] { &self.ids }
    pub(crate) fn project_zero_order(&self) -> &[usize] { &self.zero_order }
"""

def materialize(parent,destination,experiment):
    parent=Path(parent);destination=Path(destination);experiment=Path(experiment)
    manifest=json.loads((experiment/'configs/parent-built-source.json').read_text())
    assert manifest['source_commit']=='61c8a00d8323d49a011d8909113a8755906323c9'
    assert not destination.exists(),'Fresh build directory required'
    for name,h in manifest['actual_BMP_sources'].items():assert sha(parent/name)==h,('Parent source changed',name)
    destination.mkdir(parents=True)
    for name in manifest['actual_BMP_sources']:
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,p)
    # Only explicitly selected new adapters/library are built. Drop unavailable,
    # unused native CLI target declarations; keep every library/dependency byte.
    cargo=(destination/'Cargo.toml').read_text();cargo,count=re.subn(r'\n\[\[bin\]\]\nname = "(?:ciff2bmp|search)"\npath = "[^"]+"\n','',cargo);assert count==2
    (destination/'Cargo.toml').write_text(cargo,encoding='utf-8',newline='\n')
    lib=destination/'src/lib.rs';lib.write_bytes(lib.read_bytes()+b'\npub mod project_u32;\n')
    search=destination/'src/search.rs';original=search.read_text();needle='        Self { ids, zero_order }\n    }\n}'
    assert original.count(needle)==1;changed=original.replace(needle,needle[:-1]+ACCESSORS+'}')
    assert changed.replace(ACCESSORS,'')==original;search.write_text(changed,encoding='utf-8',newline='\n')
    additions={'guarded_u32.rs':'src/project_u32.rs','project_benchmark_u64.rs':'src/bin/project_benchmark_u64.rs','project_benchmark_u32.rs':'src/bin/project_benchmark_u32.rs'}
    for source,target in additions.items():shutil.copyfile(experiment/'code'/source,destination/target)
    hashes={p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}
    assert sha(destination/'Cargo.lock')==manifest['actual_BMP_sources']['Cargo.lock']
    for name,h in manifest['actual_BMP_sources'].items():
        if name not in ['Cargo.toml','src/lib.rs','src/search.rs']:assert hashes[name]==h
    receipt={'status':'MATERIALIZED_NO_QUERYEXECUTION','parent_backend_source':manifest['source_commit'],'native_revision':manifest['BMP_upstream_revision'],'parent_source_files':manifest['actual_BMP_sources'],'files':hashes,'deltas':['Add guarded U32 module and two compiled-width resident adapters','Add read-only CanonicalContext getters;all original U64 search bodies unchanged','Remove unused nativeCLI target declarations from materialized Cargo only;lock/dependencies unchanged','No index writes or conversions']}
    (destination/'materialization.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n');return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--experiment',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();v=materialize(a.parent,a.destination,a.experiment);print(json.dumps({'status':v['status'],'files':len(v['files'])}))
if __name__=='__main__':main()
