from pathlib import Path
import struct,numpy as np

def make_fixture(root):
    ids=[2**64-1,7,0,3,20];rows=[{0:255,2:1},{0:255,2:1},{},{1:16},{2:2}]
    terms=[];weights=[];offsets=[0]
    for row in rows:
        for t,w in sorted(row.items()):terms.append(t);weights.append(w)
        offsets.append(len(terms))
    gsp=root/'fixture.gsp';query=root/'fixture.tsv';dictionary=b''.join(struct.pack('<I',2)+('t%d'%t).encode() for t in range(3))
    gsp.write_bytes(b'GSPIDX01'+struct.pack('<IIQ',len(ids),3,len(terms))+struct.pack('<5Q',*ids)+struct.pack('<6Q',*offsets)+struct.pack('<%dI'%len(terms),*terms)+bytes(weights)+dictionary)
    query.write_text('empty\ntie 0:32 2:1\n')
    expected={}
    for qid,q in [('empty',{}),('tie',{0:32,2:1})]:
        expected[qid]=sorted([(identifier,sum(q.get(t,0)*w for t,w in row.items())) for identifier,row in zip(ids,rows)],key=lambda x:(-x[1],x[0]))
    return gsp,query,expected

def check_returned_scores(index,queries,tops):
    with Path(index).open('rb') as f:
        if f.read(8)!=b'GSPIDX01':raise ValueError('GSP magic')
        n,v,nnz=struct.unpack('<IIQ',f.read(16))
    ids=np.memmap(index,dtype='<u8',mode='r',offset=24,shape=n);offsets=np.memmap(index,dtype='<u8',mode='r',offset=24+8*n,shape=n+1)
    start=24+8*n+8*(n+1);terms=np.memmap(index,dtype='<u4',mode='r',offset=start,shape=nnz);weights=np.memmap(index,dtype='u1',mode='r',offset=start+4*nnz,shape=nnz)
    order=np.argsort(ids);sorted_ids=ids[order];checked=0
    for qid,values in tops.items():
        q=queries[qid]
        for identifier,score in values:
            where=int(np.searchsorted(sorted_ids,np.uint64(identifier)))
            if where>=n or int(sorted_ids[where])!=identifier:raise ValueError('returned ID outside corpus')
            d=int(order[where]);lo,hi=int(offsets[d]),int(offsets[d+1])
            actual=sum(q.get(int(t),0)*int(w) for t,w in zip(terms[lo:hi],weights[lo:hi]))
            if actual!=score:raise ValueError('independent returned-score mismatch')
            checked+=1
    return checked
