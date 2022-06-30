import sys
import os
import torch
from pathlib import Path
from .base import Featurizer

MODEL_CACHE_DIR = Path("./models")
os.makedirs(MODEL_CACHE_DIR,exist_ok=True)

class BeplerBergerFeaturizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("BeplerBerger", 6165, save_dir)
        
        from dscript.language_model import lm_embed
        
        self._max_len = 800
        self._embed = lm_embed

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]
        
        lm_emb = self._embed(seq, use_cuda=self.on_cuda)
        return lm_emb.squeeze().mean(0)
    
class ESMFeaturizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("ESM", 1280, save_dir)
        
        import esm
        torch.hub.set_dir(MODEL_CACHE_DIR)

        self._max_len = 1024

        self._esm_model, self._esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter([('sequence',seq)])
        batch_tokens = batch_tokens.to(self.device)
        results = self._esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        tokens = token_representations[0, 1 : len(seq) + 1]

        return tokens.mean(0)
        
class ProseFeaturizer(Featurizer):
        def __init__(self,
                     save_dir: Path = Path().absolute()
                    ):
            super().__init__("Prose", 6165, save_dir)
            
            from prose.alphabets import Uniprot21
            from prose.models.multitask import ProSEMT
            
            self._max_len = 800
            self._prose_model = ProSEMT.load_pretrained(path=f"{MODEL_CACHE_DIR}/prose_mt_3x1024.sav")
            
            self._register_cuda("model", self._prose_model)
            
            self._prose_alphabet = Uniprot21()

        def _transform(self, seq):
            if len(seq) > self._max_len:
                seq = seq[:self._max_len]

            x = seq.upper().encode('utf-8')
            x = self._prose_alphabet.encode(x)
            x = torch.from_numpy(x)
            x = x.to(self.device)
            x = x.long().unsqueeze(0)
            
            z = self._prose_model.transform(x)
            z = z.squeeze(0)

            return z.mean(axis=0)


class ProtBertFeaturizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("ProtBert", 1024, save_dir)
        
        from transformers import AutoTokenizer, AutoModel, pipeline
        self._max_len = 1024

        self._protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers")
        self._protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert",cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers")
        self._protbert_feat = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer)
        
        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline
        if device.type == 'cpu':
            d = -1
        else:
            d = device.index

        pipe = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer, device=d)
        self._protbert_feat = pipe
        return pipe
    
    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        embedding = torch.tensor(self._protbert_feat(self._space_sequence(seq)))
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len+1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        return feats.mean(0)
        
class ProtT5XLUniref50Featurizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("ProtT5XLUniref50", 1024, save_dir)

        self._max_len = 1024
        
        self._protbert_model, self._protbert_tokenizer = ProtT5XLUniref50Featurizer._get_T5_model()
        self._register_cuda("model", self._protbert_model)

    @staticmethod    
    def _get_T5_model():
        from transformers import T5Tokenizer, T5EncoderModel
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                               cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers"
                                              )
        model = model.eval() # set model to evaluation model
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                                do_lower_case=False,
                                                cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers"
                                               ) 

        return model, tokenizer
    
    @staticmethod
    def _space_sequence(x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        token_encoding = self._protbert_tokenizer.batch_encode_plus(ProtT5XLUniref50Featurizer._space_sequence(seq),
                                                     add_special_tokens=True,
                                                     padding="longest"
                                                    )
        input_ids      = torch.tensor(token_encoding['input_ids'])
        attention_mask = torch.tensor(token_encoding['attention_mask'])

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            embedding = self._protbert_model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state
            seq_len = len(seq)
            start_Idx = 1
            end_Idx = seq_len+1
            seq_emb = embedding[0][start_Idx:end_Idx]

        return seq_emb.mean(0)
        
class BindPredict21Featurizer(Featurizer):
    def __init__(self,
                 save_dir: Path = Path().absolute()
                ):
        super().__init__("BindPredict21", 128, save_dir)

        BINDPREDICT_DIR = '/afs/csail.mit.edu/u/s/samsl/Work/Applications/bindPredict'
        model_prefix = f'{BINDPREDICT_DIR}/trained_models/checkpoint'
        sys.path.append(BINDPREDICT_DIR)
        from architectures import CNN2Layers
        
        self._max_len = 1024
        
        self._p5tf = ProtT5XLUniref50Featurizer(save_dir = save_dir)
        self._md = CNN2Layers(1024,128,5,1,2,0)
        self._md.load_state_dict(torch.load(f"{model_prefix}5.pt", map_location='cuda:1')['state_dict'])
        self._md = self._md.eval()
        self._cnn_first = self._md.conv1[:2]
        
        self._register_cuda("pt5_featurizer", self._p5tf, lambda x,d: x.cuda(d))
        self._register_cuda("model", self._md)
        self._register_cuda("cnn", self._cnn_first)

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        protbert_e = self._p5tf(seq)
        bindpredict_e = self._cnn_first(protbert_e.view(1,1024,-1))
        return bindpredict_e.mean(axis=2).squeeze()