"""Independent full GSP structure and CIFF document-ID replay (CPU only).

Posting bodies are not independently replayed; report that limit explicitly.
NumPy is used in bounded chunks, without loading Python objects per posting.
"""
import hashlib,io,json,struct
from pathlib import Path
import numpy as np

def varint(f):
    value=0
    for shift in range(0,70,7):
        b=f.read(1)
        if not b:raise ValueError('truncated varint')
        x=b[0]
        if shift==63 and x>1:raise ValueError('varint overflow')
        value|=(x&127)<<shift
        if x<128:return value
    raise ValueError('overlong varint')

def skip(f,wire,end):
    if wire==0:varint(f)
    elif wire in (1,2,5):
        n=varint(f) if wire==2 else (8 if wire==1 else 4)
        if n>end-f.tell():raise ValueError('field outside record')
        f.seek(n,1)
    else:raise ValueError('unsupported protobuf wire')
    if f.tell()>end:raise ValueError('record overflow')

def validate(gsp,ciff,metadata,chunk=1048576):
    gsp=Path(gsp);ciff=Path(ciff)
    with gsp.open('rb') as f:
        if f.read(8)!=b'GSPIDX01':raise ValueError('GSP magic')
        header=f.read(16)
        if len(header)!=16:raise ValueError('GSP header truncated')
        n,v,nnz=struct.unpack('<IIQ',header)
        if n!=metadata['docs'] or v!=metadata['vocab'] or nnz>metadata['declared_source_postings']:raise ValueError('GSP dimensions')
        if not n or not v:raise ValueError('empty corpus/vocabulary')
        dictionary_offset=24+8*n+8*(n+1)+5*nnz
        if dictionary_offset>gsp.stat().st_size:raise ValueError('GSP arrays truncated')
        f.seek(dictionary_offset);digest=hashlib.sha256()
        for _ in range(v):
            raw=f.read(4)
            if len(raw)!=4:raise ValueError('dictionary length truncated')
            length=struct.unpack('<I',raw)[0]
            if not 0<length<1048576:raise ValueError('invalid dictionary length')
            term=f.read(length)
            if len(term)!=length:raise ValueError('dictionary truncated')
            digest.update(raw);digest.update(term)
        if f.read(1):raise ValueError('GSP trailing bytes')
    if digest.hexdigest()!=metadata['dictionary_sha256']:raise ValueError('dictionary identity')
    ids=np.memmap(gsp,dtype='<u8',mode='r',offset=24,shape=(n,))
    offsets=np.memmap(gsp,dtype='<u8',mode='r',offset=24+8*n,shape=(n+1,))
    term_offset=24+8*n+8*(n+1)
    terms=np.memmap(gsp,dtype='<u4',mode='r',offset=term_offset,shape=(nnz,)) if nnz else np.empty(0,dtype='<u4')
    impacts=np.memmap(gsp,dtype='u1',mode='r',offset=term_offset+4*nnz,shape=(nnz,)) if nnz else np.empty(0,dtype='u1')
    if int(offsets[0])!=0 or int(offsets[-1])!=nnz:raise ValueError('CSR endpoints')
    if len(np.unique(ids))!=n:raise ValueError('duplicate original passage ID')
    for start in range(0,n,chunk):
        end=min(n,start+chunk)
        if np.any(offsets[start+1:end+1]<offsets[start:end]):raise ValueError('nonmonotonic CSR offsets')
    max_impact=0
    for start in range(0,nnz,chunk):
        end=min(nnz,start+chunk)
        if np.any(terms[start:end]>=v) or np.any(impacts[start:end]==0):raise ValueError('term domain or retained zero impact')
        max_impact=max(max_impact,int(impacts[start:end].max()))
    for start in range(1,nnz,chunk):
        end=min(nnz,start+chunk)
        bad=np.flatnonzero(terms[start:end]<=terms[start-1:end-1])+start
        if len(bad):
            boundaries=np.searchsorted(offsets,bad)
            if np.any(boundaries>=len(offsets)) or np.any(offsets[boundaries]!=bad):raise ValueError('duplicate or unordered per-document term')
    with ciff.open('rb') as f:
        f.seek(metadata['posting_section_end'])
        for d in range(n):
            length=varint(f)
            if length>1048576:raise ValueError('unreasonable document record')
            raw=f.read(length)
            if len(raw)!=length:raise ValueError('truncated document record')
            b=io.BytesIO(raw);position=0;original=None
            while b.tell()<len(raw):
                key=varint(b)
                if not key:raise ValueError('zero protobuf field')
                if key==8:position=varint(b)
                elif key==18:
                    size=varint(b)
                    if size>len(raw)-b.tell():raise ValueError('document ID length')
                    value=b.read(size)
                    if not value or not value.isdigit():raise ValueError('nonnumeric original ID')
                    original=int(value)
                else:skip(b,key&7,len(raw))
            if position!=d or original!=int(ids[d]):raise ValueError('CIFF/GSP original ID or order mismatch at document '+str(d))
        if f.read(1):raise ValueError('CIFF trailing document records')
    return {'success':True,'docs':n,'vocab':v,'nnz':nnz,'dictionary_sha256':digest.hexdigest(),'all_document_ids_replayed':n,'max_impact':max_impact,'maximum_term_document_multiplicity':1 if nnz else 0,
      'qualification':'Every GSP offset/term/impact and original document ID checked; all CIFF DocRecords independently replayed. Full source posting bodies are not independently replayed. Importer df/delta assertions and independent import fixtures provide separate conversion evidence. No query retrieval or performance claim.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('gsp');p.add_argument('ciff');p.add_argument('metadata');p.add_argument('output');a=p.parse_args()
    result=validate(a.gsp,a.ciff,json.loads(Path(a.metadata).read_text()))
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['success','docs','nnz','all_document_ids_replayed']}))
