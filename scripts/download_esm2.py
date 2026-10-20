"""Download one immutable Hugging Face revision and record the resolved backbone."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from diffcl.common import sha256,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='models/esm2_650m')
    p.add_argument('--revision',default='main');a=p.parse_args()
    model='facebook/esm2_t33_650M_UR50D';api=HfApi();info=api.model_info(model,revision=a.revision)
    names={x.rfilename for x in info.siblings}
    weight='*.safetensors' if any(x.endswith('.safetensors') for x in names) else 'pytorch_model*.bin'
    snapshot_download(model,revision=info.sha,local_dir=a.output,allow_patterns=['*.json','vocab.txt',weight])
    root=Path(a.output)
    write_json(root/'backbone_manifest.json',{'model':model,'requested_revision':a.revision,'resolved_revision':info.sha,
               'sha256':{x.name:sha256(x) for x in sorted(root.iterdir()) if x.is_file() and x.name!='backbone_manifest.json'}})
    print(f'Backbone downloaded at immutable revision {info.sha}')
