"""Independent CIFF-to-GSP checks; imports only, no retrieval or timing."""
import importlib.util,struct,subprocess,sys,tempfile
from pathlib import Path

def check(executable,repository):
    spec=importlib.util.spec_from_file_location('cpu_fixture_helpers',repository/'tests/check_cpu_pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=[(1<<64)-1,7,0,1234567890123,2]
    docs=[{0:255,2:1},{0:0,1:16},{},{1:32},{0:2,2:0}]
    with tempfile.TemporaryDirectory(prefix='full-import-fixture-') as directory:
        root=Path(directory);ciff=root/'fixture.ciff'
        with ciff.open('wb') as f:
            f.write(module.frame(module.integer(1,1)+module.integer(2,3)+module.integer(3,len(ids))))
            for t in range(3):
                postings=[(d,row[t]) for d,row in enumerate(docs) if t in row]
                message=module.data(1,('t%d'%t).encode())+module.integer(2,len(postings));previous=0
                for d,w in postings:
                    message+=module.data(4,module.integer(1,d-previous)+module.integer(2,w));previous=d
                f.write(module.frame(message))
            for d,identifier in enumerate(ids):f.write(module.frame(module.integer(1,d)+module.data(2,str(identifier).encode())))
        for count in (1,3,len(ids)):
            out=root/('n%d.gsp'%count)
            subprocess.run([str(executable),'import',str(ciff),str(out),str(count)],check=True)
            data=out.read_bytes();assert data[:8]==b'GSPIDX01'
            n,v,nnz=struct.unpack_from('<IIQ',data,8);assert n==count and v==3
            chosen=set(sorted(ids,key=lambda x:(module.mix(x^0x47505532303236),x))[:count])
            expected_ids=[x for x in ids if x in chosen]
            got_ids=list(struct.unpack_from('<%dQ'%n,data,24));assert got_ids==expected_ids
            pos=24+8*n;offsets=struct.unpack_from('<%dQ'%(n+1),data,pos);pos+=8*(n+1)
            terms=struct.unpack_from('<%dI'%nnz,data,pos);pos+=4*nnz;weights=data[pos:pos+nnz];pos+=nnz
            expected_nnz=0
            for i,identifier in enumerate(got_ids):
                expected={t:w for t,w in docs[ids.index(identifier)].items() if w}
                pairs=list(zip(terms[offsets[i]:offsets[i+1]],weights[offsets[i]:offsets[i+1]]))
                assert len(pairs)==len(expected) and dict(pairs)==expected
                expected_nnz+=len(expected)
            assert offsets[0]==0 and offsets[-1]==nnz==expected_nnz
            for t in range(3):
                length=struct.unpack_from('<I',data,pos)[0];pos+=4
                assert data[pos:pos+length]==('t%d'%t).encode();pos+=length
            assert pos==len(data)
            before=out.read_bytes();result=subprocess.run([str(executable),'import',str(ciff),str(out),str(count)],capture_output=True)
            assert result.returncode!=0 and out.read_bytes()==before,'existing output overwritten'
        for count in (0,len(ids)+1):
            result=subprocess.run([str(executable),'import',str(ciff),str(root/('bad%d.gsp'%count)),str(count)],capture_output=True)
            assert result.returncode!=0
    print('PASS: full/subset membership, original uint64 IDs/order, delta decoding, zero-impact filtering, dictionary, overwrite/domain rejection')

if __name__=='__main__':check(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
