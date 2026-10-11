#!/usr/bin/env python3
"""Check frozen October IEEE Access numeric tables against recovered evidence.

With --tex NEW/main.tex, additionally detect any changed numeric row in the
manuscript. This does not evaluate figure authenticity, and
cannot reproduce upstream inference without model/dataset artifacts.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: str) -> list[dict]:
    with (ROOT/path).open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def extract_table_rows(src: str | Path, label: str) -> list[str]:
    text = Path(src).read_text(encoding='utf8')
    match = re.search(r'\\label\{' + re.escape(label) + r'\}', text)
    if not match:
        raise AssertionError(f'Numeric table label {label} missing from {src}; verify relabeling manually')
    chunk = text[match.end():]
    start = chunk.find(r'\begin{tabular')
    end = chunk.find(r'\end{tabular', start)
    if start < 0 or end < 0:
        raise AssertionError(f'Numeric table {label} is not a supported tabular environment')
    out=[]
    for line in chunk[start:end].splitlines():
        if '&' not in line or '\\\\' not in line:
            continue
        s=line.strip()
        if label=='tab:fusion' and not re.match(r'^(?:0\.|1\.00)',s):
            continue
        if label=='tab:replication' and not re.match(r'^(?:No color|With color)',s):
            continue
        if label=='tab:h1' and not re.match(r'^(?:Text2Pos|Text2Loc|&\s*(?:1|5|10)\s*&)',s):
            continue
        if label=='tab:baseline' and not re.match(r'^(?:Text2Pos|Text2Loc)',s):
            continue
        s=re.sub(r'\\citep?\{[^}]+\}','',s)
        s=s.replace(r'\textbf{','').replace('}','')
        s=s.replace(r'\%','%').replace('\\\\','')
        s=re.sub(r'\s+',' ',s).strip()
        out.append(s)
    return out


def pct(frac: float)->str:
    return f'{frac*100:.2f}'


def delta(frac:float)->str:
    return f'{frac*100:+.2f}'


def run(tex: Path | None) -> dict:
    frozen=json.loads((ROOT/'manuscript_alignment/Numerical_Table_Reference.json').read_text())
    tab5=load_csv('manuscript_alignment/FROZEN_TABLE_5.csv')
    original=load_csv('results/null_control/TABLE_source_membership_matched_null_primary.csv')
    t2=load_csv('results/text2locpp/TABLE_text2locpp_semantic_vs_matched_null.csv')
    t2source=load_csv('results/text2locpp/TABLE_text2locpp_semantic_vs_source.csv')
    nine=load_csv('results/text2locpp/TABLE_nine_comparison_sensitivity.csv')
    original={(r['model'],int(r['K'])):r for r in original if r['subset']=='V>1'}
    t2={int(r['K']):r for r in t2}
    t2source={int(r['K']):r for r in t2source}
    nine={(r['model'].lower(),int(r['K'])):r for r in nine}
    assert len(tab5)==9, 'Table 5 must contain nine rows'
    for rr in tab5:
        model=rr['model'];k=int(rr['K']);ci=nine[model,k]
        if model=='text2loc++':
            source=float(t2source[k]['source_recall'])
            semantic=float(t2[k]['semantic_recall'])
            null=float(t2[k]['corrected_null_recall'])
        else:
            r=original[model,k]
            source=float(r['source_recall'])
            semantic=float(r['semantic_recall'])
            null=float(r['corrected_null_recall_mean'])
        actual={
            'source_pct':pct(source),
            'semantic_pct':pct(semantic),
            'semantic_minus_source_pp':delta(semantic-source),
            'matched_null_pct':pct(null),
            'semantic_minus_null_pp':delta(semantic-null),
            'simultaneous_9_lower_pp':f'{float(ci["simultaneous_9_low_pp"]):.2f}',
            'simultaneous_9_upper_pp':f'{float(ci["simultaneous_9_high_pp"]):.2f}',
        }
        for field,val in actual.items():
            assert rr[field]==val, f'FROZEN TABLE 5 changed: {model} K={k} {field}: {rr[field]} != {val}'

    # Independently check recorded population statistics from original audits.
    counts=frozen['counts']
    d={(r['condition'],r['scope']):r for r in load_csv('results/heldout/TABLE_valid_set_distribution.csv')}
    d=d['primary_color','pooled']
    assert int(d['n'])==counts['heldout_queries']
    assert int(d['V0'])==counts['heldout_V0']
    assert int(d['V1'])==counts['heldout_V1']
    assert int(d['n'])-int(d['V0'])-int(d['V1'])==counts['ambiguous_queries']
    ident=next(r for r in load_csv('results/heldout/TABLE_identifiability_bound.csv') if r['scope']=='pooled')
    for col, expected in [('distinct_exact_texts','distinct_exact_texts'),('texts_with_conflicting_sources','conflicting_text_groups'),('queries_in_conflicting_text_groups','conflicting_text_queries'),('bound_numerator','heldout_exact_bound_numerator')]:
        assert int(ident[col])==counts[expected], f'Conflicting held-out identity statistic {col}'
    pop={(r['condition'],r['metric']):r for r in load_csv('results/heldout/TABLE_overall_metrics.csv') if r['model']=='population'}
    assert int(pop['primary_color','valid_set_nonempty']['numerator'])==counts['heldout_nonempty']
    assert int(pop['primary_color','source_supported']['numerator'])==counts['heldout_source_supported']
    assert math.isclose(counts['controlled_unique_texts']/counts['controlled_evaluations']*100,0.2111,abs_tol=0.00005)
    assert counts['matched_null_realizations']==1000

    report={
        'status':'PASS',
        'numeric_tables_locked':['tab:fusion','tab:replication','tab:h1','tab:baseline'],
        'recovered_heldout_population_counts':'PASS',
        'table5_nine_original_evidence_rows':'PASS',
        'manuscript_tex_checked':str(tex) if tex else None,
        'provenance_note':'Manuscript table rows are compared when --tex is supplied.',
    }
    if tex is not None:
        assert tex.is_file(), f'LaTeX file does not exist: {tex}'
        for label, expected in frozen['table_rows'].items():
            actual=extract_table_rows(tex,label)
            if actual!=expected:
                for idx in range(max(len(actual),len(expected))):
                    old=expected[idx] if idx<len(expected) else '<missing>'
                    new=actual[idx] if idx<len(actual) else '<missing>'
                    if old!=new:
                        raise AssertionError(f'Changed {label} numeric row {idx+1}:\n  frozen: {old}\n  current: {new}')
        report['all_four_manuscript_tables_match_frozen_snapshot']='PASS'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tex',type=Path,default=None,help='Optional manuscript main.tex')
    args=p.parse_args()
    print(json.dumps(run(args.tex),indent=2))
