import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
import torch
from transformers import EsmConfig,EsmModel,EsmTokenizer
from diffcl.common import columns,components

torch.set_num_threads(1)

@pytest.fixture(scope='session')
def toy(tmp_path_factory):
    root=tmp_path_factory.mktemp('tiny_esm')
    tokens=['<cls>','<pad>','<eos>','<unk>','L','A','G','V','S','E','R','T','I','D','P','K','Q','N','F','Y','M','H','W','C','X','B','U','Z','O','.','-','<null_1>','<mask>']
    (root/'vocab.txt').write_text('\n'.join(tokens)+'\n')
    EsmTokenizer(vocab_file=str(root/'vocab.txt')).save_pretrained(root)
    EsmModel(EsmConfig(vocab_size=33,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,
                       max_position_embeddings=64,pad_token_id=1,mask_token_id=32,token_dropout=False,
                       position_embedding_type='rotary')).save_pretrained(root)
    df=pd.DataFrame([
        dict(PDB='1AAA',mutation='A:A2V',a='MAAGLA',b='GQGGKG',a_mut='MVAGLA',b_mut='GQGGKG',ddG=-1.),
        dict(PDB='1AAA',mutation='A:G4W',a='MAAGLA',b='GQGGKG',a_mut='MAAWLA',b_mut='GQGGKG',ddG=2.),
        dict(PDB='2BBB',mutation='A:R2H',a='TRYSFP',b='YLYDSE',a_mut='THYSFP',b_mut='YLYDSE',ddG=.5),
        dict(PDB='2BBB',mutation='B:D4N',a='TRYSFP',b='YLYDSE',a_mut='TRYSFP',b_mut='YLYNSE',ddG=3.)])
    seq=sorted({s for _,r in df.iterrows() for c in components(r,columns(df)) for s in c if s})
    hom=root/'homology';hom.mkdir();(hom/'sequences.fasta').write_text(''.join(f'>s{i}\n{s}\n' for i,s in enumerate(seq)))
    (hom/'all_vs_all.tsv').write_text('')
    cfg=json.loads(Path('configs/reference.json').read_text());cfg.pop('selection')
    cfg.update(model_locate=str(root),model_dim=16,hidden_size=16,num_latent_tokens=4,proj_dim=8,
               esm_chunk_overlap=8,dropout=0.,mamba_dropout=0.,batch_size=4,eval_batch_size=4,
               allow_mamba_fallback=True,strict_reviewer_mode=False,num_workers=0,
               cl_warmup_start_epoch=0,cl_warmup_end_epoch=0)
    return cfg,df,hom
