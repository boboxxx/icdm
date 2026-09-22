"""Mechanical publication checks, explicitly not a scientific or visual review."""
import json
import re
import sys
from pathlib import Path
try:
    import fitz
except ImportError:
    fitz=None
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
    from pypdf import PdfReader

root=Path(sys.argv[1])
log=(root/'main.log').read_text(errors='replace')
if fitz is not None:
    doc=fitz.open(root/'main.pdf')
    text='\n'.join(p.get_text() for p in doc)
    pages=len(doc)
else:
    doc=PdfReader(root/'main.pdf')
    text='\n'.join(p.extract_text() for p in doc.pages)
    pages=len(doc.pages)
text=re.sub(r'\s+',' ',text)
bad=[phrase for phrase in ['awaits complete','experiment is running','Working manuscript','SYNTHETIC','??'] if phrase in text]
if bad: raise ValueError('Incomplete or invalid content: '+str(bad))
if 'undefined references' in log or 'Citation' in log and 'undefined' in log:
    raise ValueError('Unresolved citations/references')
overflows=[float(s) for s in re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)',log)]
if any(x>2 for x in overflows): raise ValueError('Layout overflow beyond 2 pt')
if pages>7: raise ValueError('Exceeds VTC extended page limit')
if fitz is not None:
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(root/f'page_{i+1}.png')
print(json.dumps({'status':'mechanical_checks_passed','pages':pages,'overflows_pt':overflows,
                  'author_placeholder':'Author information pending' in text,
                  'scientific_and_visual_final_review':'not performed by this script'}))
