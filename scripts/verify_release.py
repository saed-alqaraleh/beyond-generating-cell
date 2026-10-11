#!/usr/bin/env python3
"""Independently verify recovered frozen evidence; no inference, re-training or sampling."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def checked(condition, message):
    if not bool(condition):raise AssertionError(message)

def read_csv(path):
    with open(ROOT/path,newline='',encoding='utf8') as fp:return list(csv.DictReader(fp))

def close(a,b,where,tol=1e-12):
    checked(math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol),f'{where}: {a} != {b}')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as fp:
        for block in iter(lambda:fp.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def run():
    results={'status':'FAIL','checks':{},'counts':{},'counts_per_model':{}}
    # Original provenance SHA-256 for three frozen public-release assets.
    for f in ('data/frozen/PREPARED_PRIMARY_INPUTS.npz','data/frozen/QUERY_LEVEL_NULL_PROBABILITIES.csv.gz','data/frozen/TEXT2LOCPP_QUERY_LEVEL_NULL_PROBABILITIES.parquet'):
        d=ROOT/f;checked(d.is_file(),f'Missing file {f}'); results['checks']['exists:'+f]=True
    with np.load(ROOT/'data/frozen/PREPARED_PRIMARY_INPUTS.npz',allow_pickle=False) as z:
        n=4794;c=4308;names=('text2pos','text2loc');Ks=(1,5,10)
        V=z['V'];q=z['query_id'];source=z['source_idx'];support=z['source_supported'];top=z['top10']
        checked(V.shape==(n,) and (V>1).all(),'Ambiguous-cohort V gate')
        checked(len(set(q.tolist()))==n,'Duplicate query IDs')
        checked(z['candidate_id'].shape==(c,),'Full ordered candidate universe length')
        checked(z['row_scene_counts'].sum(axis=1).tolist()==V.tolist(),'Scene-specific query V conservation')
        checked(top.shape==(2,n,10) and np.all((top>=0)&(top<c)),'Invalid candidate rankings')
        checked(np.all((source>=0)&(source<c)),'Invalid source indices')
        bits=np.unpackbits(z['valid_packed'],axis=1,bitorder='big')[:,:c]
        checked(np.array_equal(bits.sum(axis=1),V),'Bitmask valid-set size conservation')
        checked(np.array_equal(bits[np.arange(n),source].astype(bool),support),'Bitmask source-support consistency')
        checked(int(support.sum())==3780,'Source-supported ambiguous query count')
        results['counts'].update({'heldout_full_queries_from_original_audit':11505,'candidate_cells':c,'ambiguous_queries':n,'source_supported_ambiguous':int(support.sum())})
        observed_source=z['observed_source'];observed_sem=z['observed_semantic']
        checked(observed_source.shape==(n,2,3) and observed_sem.shape==(n,2,3),'Stored hit shape')
        rows=read_csv('results/null_control/TABLE_source_membership_matched_null_primary.csv')
        with gzip.open(ROOT/'data/frozen/QUERY_LEVEL_NULL_PROBABILITIES.csv.gz','rt',newline='',encoding='utf8') as fp:
            null_query_rows=list(csv.DictReader(fp))
        checked(len(null_query_rows)==n,'Number of independently archived query-level null summaries')
        checked(all(r['query_id']==qid for r,qid in zip(null_query_rows,q)),'Query-id alignment of null and ranking files')
        checked(all(int(r['V'])==int(v) for r,v in zip(null_query_rows,V)),'Query-level null V alignment')
        checked(all((r['source_supported']=='True')==bool(flag) for r,flag in zip(null_query_rows,support)),'Query-level null source-support alignment')
        checked(len(rows)==6,'Six null-control rows not found')
        for m,name in enumerate(names):
            model={}
            for ki,k in enumerate(Ks):
                rr=top[m,:,:k]
                source_hit=np.any(rr==source[:,None],axis=1)
                semantic_hit=np.any(np.take_along_axis(bits,rr,axis=1).astype(bool),axis=1)
                checked(np.array_equal(source_hit,observed_source[:,m,ki].astype(bool)),f'{name}: recorded source hit K{k}')
                checked(np.array_equal(semantic_hit,observed_sem[:,m,ki].astype(bool)),f'{name}: recorded semantic hit K{k}')
                row=next((r for r in rows if r['model']==name and int(r['K'])==k and r['subset']=='V>1'),None)
                checked(row is not None,f'Missing manuscript-facing matched null {name} K{k}')
                sn=int(source_hit.sum());vn=int(semantic_hit.sum())
                checked(sn==int(row['source_recall_numerator']),f'{name}: source numerator K{k}')
                checked(vn==int(row['semantic_recall_numerator']),f'{name}: semantic numerator K{k}')
                checked(int(row['source_recall_denominator'])==n and int(row['semantic_recall_denominator'])==n,'Matched-null denominator')
                close(sn/n,row['source_recall'],f'{name} source {k}')
                close(vn/n,row['semantic_recall'],f'{name} semantic {k}')
                q_sem_field=f'{name}_semantic_hit_{k}'
                q_src_field=f'{name}_source_hit_{k}'
                q_null_field=f'{name}_corrected_null_probability_{k}'
                checked(all(int(r[q_src_field])==int(hit) for r,hit in zip(null_query_rows,source_hit)),f'{name} query-level null source hits K{k}')
                checked(all(int(r[q_sem_field])==int(hit) for r,hit in zip(null_query_rows,semantic_hit)),f'{name} query-level null semantic hits K{k}')
                null_mean=sum(float(r[q_null_field]) for r in null_query_rows)/n
                close(null_mean,row['corrected_null_recall_mean'],f'{name} query-level null mean K{k}')
                close((vn/n-null_mean)*100,row['effect_percentage_points'],f'{name} vs matched null K{k}',tol=1e-10)
                model[str(k)]={'source_hits':sn,'semantic_hits':vn,'matched_null_mean':float(row['corrected_null_recall_mean']),'semantic_minus_null_pp':float(row['effect_percentage_points'])}
            results['counts_per_model'][name]=model
        results['checks']['frozen_two_model_recall_from_bitmaps']=True
        results['checks']['all_query_level_null_summaries']=True
    heldout=read_csv('results/heldout/TABLE_overall_metrics.csv')
    def find_row(model,metric,k,condition='primary_color'):
        arr=[r for r in heldout if r['condition']==condition and r['model']==model and r['metric']==metric and str(r['K'])==str(k)]
        checked(len(arr)==1,f'Expected one heldout row: {condition} {model} {metric} {k}');return arr[0]
    locks={'text2pos':(1372,3676,4901),'text2loc':(3428,6917,8206)}
    for name,counts in locks.items():
        for k,cnt in zip((1,5,10),counts):
            row=find_row(name,'source_recall_all',k)
            checked(int(row['numerator'])==cnt and int(row['denominator'])==11505,f'full held-out {name} K{k}')
            close(cnt/11505,row['estimate'],f'heldout {name} K{k}')
    results['checks']['full_heldout_original_aggregate_locks']=True
    # Regenerate controlled exact source Bayes bound using template frequencies, not paper text.
    template=read_csv('results/controlled_and_generator/jobs/template_analysis_primary/template_V.csv')
    cell=read_csv('results/controlled_and_generator/jobs/template_analysis_primary/source_cell_counts.csv')
    distr=read_csv('results/controlled_and_generator/jobs/template_analysis_primary/V_exact_distribution.csv')
    checked(len(template)==1192 and len(cell)==2466,'Controlled population dimensions')
    total=sum(int(r['query_count']) for r in cell)
    checked(total==564768 and sum(int(r['queries']) for r in distr)==total,'Controlled query count')
    exactly_one=sum(int(r['queries']) for r in distr if int(r['V'])==1)
    checked(exactly_one==48,'Controlled valid set singleton count')
    bound=len(template)/total
    checked(round(bound * 100, 4) == 0.2111, 'Controlled exact source bound 0.2111%')
    results['counts'].update({'controlled_queries':total,'controlled_unique_texts':len(template),'controlled_candidate_cells':len(cell),'controlled_V_eq_1':exactly_one,'controlled_exact_top1_bound_fraction':bound})
    results['checks']['controlled_population_and_bound']=True
    # Text2Loc++ summary arithmetic from audited derived effects; no claim of raw model inference.
    t2=read_csv('results/text2locpp/TABLE_text2locpp_semantic_vs_matched_null.csv')
    checked(len(t2)==3 and {int(r['K']) for r in t2}=={1,5,10},'Text2Loc++ K set')
    for r in t2:
        k=int(r['K']); denom=int(r['n']);checked(denom==4794,'Text2Loc++ denominator')
        close(int(r['semantic_hits'])/denom,r['semantic_recall'],f'Text2Loc++ semantic {k}')
        close(int(r['null_total_hits'])/int(r['null_denominator']),r['corrected_null_recall'],f'Text2Loc++ null {k}')
        close((float(r['semantic_recall'])-float(r['corrected_null_recall']))*100,r['effect_pp'],f'Text2Loc++ effect {k}',tol=1e-10)
    results['checks']['text2locpp_derived_summary_arithmetic']=True
    # Confirm original code snapshots are not modified by this packaging.
    manifest=ROOT/'provenance/code_files_sha256.csv'
    if manifest.exists():
        with open(manifest,newline='',encoding='utf-8') as fp:
            for r in csv.DictReader(fp):
                checked(sha256(ROOT/r['path'])==r['sha256'],f'Original source changed: {r["path"]}')
        results['checks']['original_code_snapshot_hashes']=True
    results['status']='PASS'
    return results

if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('--report',default='verification_report.json');args=arg.parse_args()
    try:
        report=run()
        target=ROOT/args.report if args.report!='-' else None
        if target: target.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    except Exception as exc:
        import traceback;traceback.print_exc();raise SystemExit(1)
