"""Build the English report from the English or Russian Markdown source."""
from pathlib import Path
import re,json,sys
ROOT=Path(__file__).resolve().parents[1]
ru='--ru' in sys.argv
source=ROOT/('latex/report_ru.md' if ru else 'latex/report_en.md')
s=source.read_text().split('<!-- Figure-source map')[0].split('<!-- Карта источников')[0]
if ru:
    s=s.replace('## Приложение А.','## Appendix A.').replace('## Приложение Б.','## Appendix B.').replace('### А.','### A.').replace('### Б.','### B.')
manifest=json.loads((ROOT/'results/data/report3/figure_manifest.json').read_text())
order=manifest['figure_order'];assert len(order)==16

def text(s):
    parts=re.split(r'(\\\(.*?\\\)|https?://[^\s]+)',s)
    out=[]
    for part in parts:
        if part.startswith(r'\('):out.append(part);continue
        if part.startswith('http'):out.append(r'\url{'+part+'}');continue
        part=part.replace('–','--').replace('—','---').replace('’',"'").replace('‘',"'").replace('“','``').replace('”',"''").replace('µ',r'\ensuremath{\mu}').replace('²',r'\textsuperscript{2}').replace('⁻',r'\textsuperscript{-}').replace('¹',r'\textsuperscript{1}')
        part=part.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
        part=re.sub(r'\*\*(.+?)\*\*',r'\\textbf{\1}',part);part=re.sub(r'(?<!\*)\*([^*]+)\*',r'\\emph{\1}',part)
        part=re.sub(r'\[(\d+(?:\s*,\s*\d+)*)\]',lambda m:r'\cite{'+','.join('ref'+n.strip() for n in m[1].split(','))+'}',part)
        part=re.sub(r'Figs\. (\d+)--(\d+)',r'Figs.~\\ref{fig:\1}--\\ref{fig:\2}',part)
        part=re.sub(r'Fig\. (\d+)',r'Fig.~\\ref{fig:\1}',part)
        part=re.sub(r'Table 1\b',r'Table~\\ref{tab:sampling}',part)
        out.append(part)
    return ''.join(out)

old=(ROOT/'latex/report_en.tex').read_text();preamble=old.split(r'\begin{document}')[0]
preamble=preamble.split(r'\newcommand{\reportfigure}')[0].split(r'\usepackage{longtable}')[0]
preamble=preamble.replace(r'\graphicspath{{../results/figures/report/}}',r'\graphicspath{{../results/figures/report3/}}')
preamble=preamble.replace('Reservoir and spin dynamics of a polariton condensate','Reservoir, spin and polarization dynamics of a polariton condensate')
preamble=preamble.rstrip()+r'''
\usepackage{longtable}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}
\title{Reservoir, spin and polarization dynamics of a polariton condensate}
\author{Korostelev Nikita}
\date{}
\begin{document}
\maketitle
\clearpage
'''
if ru:
    preamble=preamble.replace('[T1]','[T2A]').replace('[english]{babel}','[english,russian]{babel}').replace('Reservoir, spin and polarization dynamics of a polariton condensate','Динамика резервуаров, спина и поляризации поляритонного конденсата').replace('Korostelev Nikita','Коростелев Никита')
lines=s.splitlines();i=0;output=[];listed=False;appendix=False;table_caption=None;bib=False
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line:output.append('');continue
    if line.startswith('# ') or line in ('**Korostelev Nikita**','**Коростелев Никита**') or line.startswith('> ') or line.startswith('<!--'):continue
    if line==r'\[':
        eq=[]
        while i<len(lines) and lines[i].strip()!=r'\]':eq.append(lines[i]);i+=1
        i+=1;eq='\n'.join(eq)
        tag=re.search(r'\\tag\{(\d+)\}',eq)
        if tag:eq=eq.replace(tag[0],tag[0]+r'\label{eq:'+tag[1]+'}')
        if tag and tag[1]=='1':
            eq=eq.replace(r'\qquad\n',r'\qquad\n')
            eq=eq.replace(r',\qquad\n',r',\qquad\n')
            eq=eq.replace('\\qquad\n\\partial_t', '\\\\\n\\partial_t')
            tagpos=eq.index(r'\tag');eq=r'\begin{gathered}'+'\n'+eq[:tagpos]+r'\end{gathered}'+'\n'+eq[tagpos:]
        output.extend([r'\[',eq,r'\]']);continue
    if line.startswith('## '):
        if listed:output.append(r'\end{itemize}');listed=False
        title=line[3:]
        if title in ('References','Литература'):output.append(r'\clearpage\begin{thebibliography}{6}');bib=True;continue
        if title.startswith('Appendix '):
            if not appendix:output.extend([r'\clearpage',r'\appendix']);appendix=True
            title=re.sub(r'^Appendix [AB]\. ','',title)
        elif re.match(r'^\d+\. ',title):title=re.sub(r'^\d+\. ','',title)
        else:output.append(r'\section*{'+text(title)+'}');continue
        output.append(r'\section{'+text(title)+'}');continue
    if line.startswith('### '):
        title=re.sub(r'^(?:\d+|[AB])\.\d+\s+','',line[4:]);output.append(r'\subsection{'+text(title)+'}');continue
    if bib:
        m=re.match(r'\[(\d+)\]\s+(.*)',line)
        if m:output.append(r'\bibitem{ref'+m[1]+'} '+text(m[2]))
        continue
    m=re.match(r'\*\*(?:Figure|Рисунок) (\d+)\.\*\*\s+(.*)',line)
    if m:
        n=int(m[1]);key=order[n-1];image=manifest['figures'][key]['file'];width='105mm' if key=='N5_TE_TM_angular_response' else r'0.96\textwidth'
        output.extend([r'\begin{figure}[H]',r'\centering',r'\includegraphics[width='+width+']{../'+image+'}',r'\caption{'+text(m[2])+'}',r'\label{fig:'+str(n)+'}',r'\end{figure}']);continue
    if line.startswith(('**Table 1.','**Таблица 1.')):
        table_caption=line.split('**',2)[2].strip();continue
    if line.startswith('|'):
        rows=[line]
        while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
        sampling=table_caption is not None
        if sampling:output.append(r'\clearpage')
        output.extend([r'\begingroup'+(r'\footnotesize' if sampling else r'\small')+r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.2}',r'\begin{longtable}{@{}p{0.24\textwidth}p{0.72\textwidth}@{}}'])
        if sampling:output.append(r'\caption{'+('Времена снимков и параметры анализа. ' if ru else 'Snapshot times and analysis settings. ')+text(table_caption)+r'}\label{tab:sampling}\\')
        output.append(r'\toprule')
        for k,row in enumerate(rows):
            if k==1:output.append(r'\midrule');continue
            cells=[x.strip() for x in row.strip('|').split('|')];assert len(cells)==2
            output.append(' & '.join(text(c) for c in cells)+r'\\')
        output.extend([r'\bottomrule',r'\end{longtable}',r'\endgroup']);table_caption=None;continue
    if line.startswith('- '):
        if not listed:output.append(r'\begin{itemize}');listed=True
        output.append(r'\item '+text(line[2:]));continue
    if listed:output.append(r'\end{itemize}');listed=False
    output.append(text(line))
if bib:output.append(r'\end{thebibliography}')
output.append(r'\end{document}')
tex=preamble+'\n'.join(output)+'\n'
if ru:
    title='Внутренняя джозефсоновская динамика в 0D-модели'
    tex=tex.replace(r'\section{'+title+'}',r'\section{\texorpdfstring{Внутренняя джозефсоновская динамика\\в 0D-модели}{'+title+'}}')
(ROOT/('latex/report_ru.tex' if ru else 'latex/report_en.tex')).write_text(tex)
print('Wrote '+('latex/report_ru.tex' if ru else 'latex/report_en.tex')+' from',source.name,'with16 figures')
