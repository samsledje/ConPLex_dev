import torch
from pathlib import Path
from .base import Featurizer

CACHE_DIR = "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/models/huggingface/transformers"

class BeplerBergerFeaturizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("BeplerBerger", 6165)
        
        from dscript.language_model import lm_embed
        
        self._max_len = 800
        self._embed = lm_embed

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq, use_cuda=self.on_cuda)
            return lm_emb.squeeze(rm *feature).mean(0)


class ProtBertFeaturizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("ProtBert", 1024)
        
        from transformers import AutoTokenizer, AutoModel, pipeline
        self._max_len = 1024

        self._protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,cache_dir=CACHE_DIR)
        self._protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert",cache_dir=CACHE_DIR)
        self._protbert_feat = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer)
        
        def _feat_to_device(device):
            if device.type == 'cpu':
                d = -1
            else:
                d = device.index
            
            pipe = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer, device=d)
            self._protbert_feat = pipe
            return pipe
        
        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, _feat_to_device)

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        with torch.no_grad():
            embedding = torch.tensor(self._protbert_feat(self._space_sequence(seq)))
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len+1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        return feats.mean(0)