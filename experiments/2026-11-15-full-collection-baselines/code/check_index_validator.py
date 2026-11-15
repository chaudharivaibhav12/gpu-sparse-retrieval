"""Small valid/corrupt index checks before full-data validation on a new host."""
import hashlib,importlib.util,struct,sys,tempfile
from pathlib import Path
from validate_index import validate

def check(repository):
    spec=importlib.util.spec_from_file_location('fixture_helpers',repository/'tests/check_cpu_pilot.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    ids=[2**64-1,0,2,999];dictionary=b''.join(struct.pack('<I',len(x))+x for x in [b'a',b'b',b'c'])
    metadata={'docs':4,'vocab':3,'declared_source_postings':5,'dictionary_sha256':hashlib.sha256(dictionary).hexdigest(),'posting_section_end':0}
    with tempfile.TemporaryDirectory(prefix='full-index-check-') as d:
        p=Path(d);gsp=p/'x.gsp';raw=p/'x.ciff'
        records=b''.join(helpers.frame(helpers.integer(1,i)+helpers.data(2,str(x).encode())) for i,x in enumerate(ids));raw.write_bytes(records)
        def data(offsets=(0,2,2,3,4),terms=(0,2,1,0),weights=(255,1,32,2),identifiers=ids):
            return b'GSPIDX01'+struct.pack('<IIQ',4,3,len(terms))+struct.pack('<4Q',*identifiers)+struct.pack('<5Q',*offsets)+struct.pack('<%dI'%len(terms),*terms)+bytes(weights)+dictionary
        valid=data();gsp.write_bytes(valid);result=validate(gsp,raw,metadata,chunk=2)
        if not result['success'] or result['nnz']!=4:raise ValueError('valid fixture rejected')
        for blob in [data(offsets=(0,2,1,3,4)),data(terms=(0,0,1,0)),data(terms=(2,0,1,0)),data(terms=(0,3,1,0)),data(weights=(255,0,32,2)),data(identifiers=[2**64-1,0,0,999]),data(identifiers=[0,2**64-1,2,999]),valid+b'x',valid[:-1]]:
            gsp.write_bytes(blob)
            try:validate(gsp,raw,metadata,chunk=2)
            except ValueError:pass
            else:raise ValueError('corrupt GSP accepted')
        gsp.write_bytes(valid)
        for bad in [records[:-1],records+b'\0',helpers.frame(b'\x0a\x01x')+records[1:]]:
            raw.write_bytes(bad)
            try:validate(gsp,raw,metadata,chunk=2)
            except ValueError:pass
            else:raise ValueError('corrupt DocRecord accepted')
        raw.write_bytes(records);gsp.write_bytes(data(offsets=(0,0,0,0,0),terms=(),weights=()))
        if validate(gsp,raw,metadata,chunk=2)['maximum_term_document_multiplicity']!=0:raise ValueError('empty postings failed')
    print('PASS: validator valid/uint64/empty/chunk-boundary and12malformed-input checks')

if __name__=='__main__':check(Path(sys.argv[1]).resolve())
