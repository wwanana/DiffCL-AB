import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from diffcl.common import metrics,validate_frame,Relations,audit_partitions,input_keys
from scripts.prepare_data import build
from prepare_splits import prepare
from aggregate_cv import aggregate

def test_metrics_and_undefined_values():
    m=metrics([1,2,3],[2,4,6]);assert m['pcc']==pytest.approx(1) and m['r2']<0
    assert metrics([1,1],[2,3])['pcc'] is None
    assert metrics([1],[2])['r2'] is None
    with pytest.raises(ValueError):metrics([np.nan],[1])
    assert metrics([1,1,2],[1,2,3])['spearman']==pytest.approx(np.sqrt(.75))

@pytest.mark.parametrize('change',[{'ddG':np.nan},{'PDB':''},{'a_mut':'A'},{'a_mut':'MAAGLA'}])
def test_bad_input_is_rejected(toy,change):
    df=toy[1].iloc[:1].copy()
    for k,v in change.items():df[k]=v
    with pytest.raises(ValueError):validate_frame(df)

def test_direct_relations_and_both_coverages(tmp_path):
    (tmp_path/'sequences.fasta').write_text('>a\nAAAA\n>b\nAAAT\n>c\nAATT\n>d\nATTT\n')
    (tmp_path/'all_vs_all.tsv').write_text('a\tb\t0.5\t0.8\t0.8\nb\tc\t0.5\t0.9\t0.9\na\td\t0.9\t0.9\t0.7\n')
    r=Relations(tmp_path)
    assert r.related('AAAA','AAAT') and not r.related('AAAA','AATT') and not r.related('AAAA','ATTT')

def test_source_processing_is_reproducible(tmp_path):
    build('data/source',tmp_path)
    for n in ['AB645','AB1101','S1131']:
        a=pd.read_csv(tmp_path/f'{n}.csv');b=pd.read_csv(f'data/processed/{n}.csv')
        pd.testing.assert_frame_equal(a,b)

@pytest.mark.parametrize('name',['AB645','AB1101','S1131'])
@pytest.mark.parametrize('protocol',['ordinary','lh30'])
def test_all_stored_folds_are_deterministic_and_leakage_checked(name,protocol,tmp_path):
    hom='data/homology/S1131' if name=='S1131' else 'data/homology/AB1101'
    new=prepare(f'data/processed/{name}.csv',protocol,tmp_path/'folds',hom)
    saved=json.loads(Path(f'data/splits/{name}/{protocol}/manifest.json').read_text())
    assert new==saved

def test_partition_label_changes_rejected(toy):
    df=toy[1];refit=df.iloc[:3].copy();refit.loc[0,'ddG']=100
    with pytest.raises(ValueError):audit_partitions(df.iloc[:2],df.iloc[2:3],refit,df.iloc[3:])

def test_oof_aggregation_requires_complete_labels(tmp_path):
    source=pd.read_csv('data/processed/AB645.csv');m=json.loads(Path('data/splits/AB645/ordinary/manifest.json').read_text())
    files=[]
    for f in m['folds']:
        df=source.iloc[f['test']];p=tmp_path/f"fold{f['fold']}.csv"
        pd.DataFrame(dict(fold=f['fold'],source_index=df.record_id,pdb_id=df.PDB,y_true=df.ddG,y_pred=df.ddG)).to_csv(p,index=False);files.append(p)
    assert aggregate('data/processed/AB645.csv',files,tmp_path/'out')['pooled']['rmse']==0
    with pytest.raises(ValueError):aggregate('data/processed/AB645.csv',files[:-1],tmp_path/'invalid')
