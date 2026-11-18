"""Conservative compiled-artifact gate; actual execution still needs correctness."""
import hashlib,re
from pathlib import Path

def byte_load_widened_to_fp32(text):
    # Triton TTIR annotates tt.load with its POINTER tensor type, not
    # necessarily an explicit result type. Bind the loaded SSA value to the
    # unsigned conversion instead of accepting an unrelated byte signature.
    loads=re.findall(r'(?m)^\s*(%[\w.$]+)\s*=\s*tt\.load[^\n]*:\s*tensor<[^>\n]*!tt\.ptr<i8>>',text)
    return any(re.search(r'arith\.uitofp\s+'+re.escape(value)+r'\s*:\s*tensor<[^>\n]*xi8>\s+to\s+tensor<[^>\n]*xf32>',text) for value in loads)

def inspect_compact(cache):
    root=Path(cache);matches=[]
    for p in root.rglob('_fast_scatter_add_kernel.ttir'):
        text=p.read_text();ptx=p.with_suffix('.ptx')
        if not ptx.is_file():raise ValueError('Missing scoring PTX for compact TTIR')
        machine=ptx.read_text()
        # Unsigned byte pointer/load and explicit unsigned-to-FP32 conversion;
        # SSA layouts vary by GPU, so match types rather than hardcoded names.
        checks=dict(byte_input='!tt.ptr<i8>' in text,
                    byte_load=byte_load_widened_to_fp32(text),
                    widen_fp32=bool(re.search(r'arith\.uitofp[^\n]*xi8>[^\n]*xf32>',text)),
                    product_fp32=bool(re.search(r'arith\.mulf[^\n]*xf32>',text)),
                    atomic_fp32=bool(re.search(r'atom[^;\n]*\.add\.f32',machine)))
        if not all(checks.values()):raise ValueError('Compact scoring compilation not established: '+str(checks))
        matches.append(dict(ttir=str(p.relative_to(root)),ttir_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ptx=str(ptx.relative_to(root)),ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),checks=checks))
    if not matches:raise ValueError('No actual compact scoring artifacts found')
    return dict(success=True,entries=matches,qualification='Compiled uint8 load/unsigned FP32 widening/FP32 product and atomic evidence. Not an instruction trace; full same-build canonical correctness is independently required.')
