"""CPU-only packed-index fixture using the existing independent decoder."""
import importlib.util,subprocess,sys,tempfile
from pathlib import Path

def check(tool,repository):
    spec=importlib.util.spec_from_file_location('range_fixture_helpers',repository/'tests/check_range_bitpack.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    n=1031;ids=[2**63+n-d for d in range(n)];docs=[]
    for d in range(n):
        row=[] if d%11==0 else [(0,d%255+1)]
        if d in (0,1023,1030):row.append((1,255))
        docs.append(row)
    with tempfile.TemporaryDirectory(prefix='full-range-input-check-') as directory:
        root=Path(directory);gsp=root/'fixture.gsp';m.write_gsp(gsp,ids,docs,5)
        coverage={'straddle':0,'one_value_blocks':0,'multi_tile_blocks':0,'empty_terms':0,'max_width':{},'narrow_ranges':0}
        for w in (1,3,64,1024):
            rbp=root/('w%d.rbp'%w)
            subprocess.run([str(tool),'build',str(gsp),str(rbp),str(w),'positions'],check=True)
            subprocess.run([str(tool),'verify',str(gsp),str(rbp)],check=True)
            m.decode_rbp(rbp,gsp,ids,docs,5,w,coverage,'positions',0)
        if not coverage['straddle'] or coverage['max_width'][1024]!=10:raise ValueError('fixture lacks required packing coverage')
    print('PASS:4index builds/verifications, independent every-posting decoder; uint64IDs, empty rows/terms, partial ranges,10bitpositions and word straddles')

if __name__=='__main__':check(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
