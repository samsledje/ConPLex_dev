import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import pickle as pk
from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
# from dpatch import PB_Embed
from torch.nn.utils.rnn import pad_sequence
import warnings

PRECOMPUTED_PROTEIN_PATH = "precomputed_proteins.pk"
FILE_DIR = os.path.dirname(os.path.realpath((__file__)))
MODEL_DIR = f"{FILE_DIR}/models"

#################################
# Sanity Check Null Featurizers #
#################################

class Random_f:
    def __init__(self, size = 1024, pool=True):
        self.use_cuda = True
        self._size = size

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        self.precomputed = True

    def _transform(self, seq):
        return torch.rand(self._size).cuda()

    def __call__(self, seq):
        return self._transform(seq)

class Null_f:
    def __init__(self, size = 1024, pool=True):
        self.use_cuda = True
        self._size = size

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        pass

    def _transform(self, seq):
        return torch.zeros(self._size).cuda()

    def __call__(self, seq):
        return self._transform(seq)

####################################
# Language Model Alone Featurizers #
####################################

class BeplerBerger_f:
    def __init__(self, pool=True):
        from dscript.language_model import lm_embed
        from dscript.pretrained import get_pretrained
        self.use_cuda = True
        self.pool = pool
        self._size = 6165
        self._max_len = 800
        self.precomputed = False

        self._embed = lm_embed

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing bepler berger protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_BeplerBerger_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq, use_cuda=self.use_cuda)
            if self.pool:
                return lm_emb.squeeze().mean(axis=0)
            else:
                return lm_emb.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

try:
    from prose.alphabets import Uniprot21
    from prose.models.multitask import ProSEMT

    class Prose_f:
        def __init__(self, pool=True):
            from dscript.language_model import lm_embed
            from dscript.pretrained import get_pretrained
            self.use_cuda = True
            self.pool = pool
            self._size = 6165
            self._max_len = 800
            self.precomputed = False

            self._prose_model = ProSEMT.load_pretrained(path=f"{MODEL_DIR}/prose_mt_3x1024.sav")
            if self.use_cuda:
                self._prose_model = self._prose_model.cuda()
            self._prose_alphabet = Uniprot21()

        def precompute(self, seqs, to_disk_path=True, from_disk=True):
            print("--- precomputing Prose protein featurizer ---")
            assert not self.precomputed
            precompute_path = f"{to_disk_path}_Prose_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
            if from_disk and os.path.exists(precompute_path):
                print("--- loading from disk ---")
                self.prot_embs = pk.load(open(precompute_path,"rb"))
            else:
                self.prot_embs = {}
                for sq in tqdm(seqs):
                    if sq in self.prot_embs:
                        continue
                    self.prot_embs[sq] = self._transform(sq)

                if to_disk_path is not None and not os.path.exists(precompute_path):
                    print(f'--- saving protein embeddings to {precompute_path} ---')
                    pk.dump(self.prot_embs, open(precompute_path,"wb+"))
            self.precomputed = True

        @lru_cache(maxsize=5000)
        def _transform(self, seq):
            if len(seq) > self._max_len:
                seq = seq[:self._max_len]

            with torch.no_grad():
                x = seq.upper().encode('utf-8')
                x = self._prose_alphabet.encode(x)
                x = torch.from_numpy(x)
                if self.use_cuda:
                    x = x.cuda()
                x = x.long().unsqueeze(0)
                z = self._prose_model.transform(x)
                z = z.squeeze(0)

                if self.pool:
                    return z.mean(axis=0)
                else:
                    return z

        def __call__(self, seq):
            if self.precomputed:
                return self.prot_embs[seq]
            else:
                return self._transform(seq)

except (ModuleNotFoundError, ImportError):
    warnings.warn("Prose not installed -- unable to use Prose family of protein featurizers")

class ESM_f:
    def __init__(self,
                 pool: bool = True,
                 dl_path: str = MODEL_DIR,
                ):
        super().__init__()
        import esm
        if dl_path is not None:
            torch.hub.set_dir(dl_path)

        self.use_cuda = True
        self.pool = pool
        self._size = 1280
        self._max_len = 1024
        self.precomputed = False

        self._esm_model, self._esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        if self.use_cuda:
            self._esm_model = self._esm_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing ESM protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ESM_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self._esm_batch_converter([('sequence',seq)])
            if self.use_cuda:
                batch_tokens = batch_tokens.cuda()
            results = self._esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            tokens = token_representations[0, 1 : len(seq) + 1]

            if self.pool:
                return tokens.mean(0)
            else:
                return tokens

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

class ProtBert_f:
    def __init__(self,
                 pool: bool = True,
                ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel, pipeline

        self.use_cuda = True
        self.pool = pool
        self._size = 1024
        self._max_len = 1024
        self.precomputed = False

        self._protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,cache_dir=f"{MODEL_DIR}/huggingface/transformers")
        self._protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert",cache_dir=f"{MODEL_DIR}/huggingface/transformers")
        if self.use_cuda:
            self._protbert_feat = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer)
        else:
            self._protbert_feat = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer)

    def _space_sequence(self, x):
        return " ".join(list(x))

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing ProtBert protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ProtBert_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        with torch.no_grad():
            embedding = torch.tensor(self._protbert_feat(self._space_sequence(seq)))
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len+1
        seq_emb = embedding.squeeze()[start_Idx:end_Idx]

        if self.pool:
            return seq_emb.mean(0)
        else:
            return seq_emb

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

def get_T5_model():
    from transformers import T5Tokenizer, T5EncoderModel
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                           cache_dir=f"{MODEL_DIR}/huggingface/transformers"
                                          )
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                            do_lower_case=False,
                                            cache_dir=f"{MODEL_DIR}/huggingface/transformers"
                                           ) 

    return model, tokenizer
        
class ProtT5_XL_Uniref50_f:
    def __init__(self,
                 pool: bool = True,
                ):
        super().__init__()
        
        self.use_cuda = True
        self.pool = pool
        self._size = 1024
        self._max_len = 1024
        self.precomputed = False
        
        self._protbert_model, self._protbert_tokenizer = get_T5_model()
        if self.use_cuda:
            self._protbert_model = self._protbert_model.cuda()
        else:
            self._protbert_model = self._protbert_model

    def _space_sequence(self, x):
        return " ".join(list(x))

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing ProtT5_XL_Uniref50 protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ProtT5_XL_Uniref50_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq: str):
        if len(seq) > self._max_len-2:
            seq = seq[:self._max_len-2]

        token_encoding = self._protbert_tokenizer.batch_encode_plus(self._space_sequence(seq),
                                                     add_special_tokens=True,
                                                     padding="longest"
                                                    )
        input_ids      = torch.tensor(token_encoding['input_ids'])
        attention_mask = torch.tensor(token_encoding['attention_mask'])

        if self.use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            embedding = self._protbert_model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state
            # seq_len = (attention_mask[0] == 1).sum()
            # seq_emb = embedding[0][:seq_len-1]
            seq_len = len(seq)
            start_Idx = 1
            end_Idx = seq_len+1
            seq_emb = embedding[0][start_Idx:end_Idx]

        if self.pool:
            return seq_emb.mean(0)
        else:
            return seq_emb

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

####################################
# PLM --> D-SCRIPT 100 Featurizers #
####################################

class BeplerBerger_DSCRIPT_f:
    def __init__(self, pool=True):
        from dscript.language_model import lm_embed
        from dscript.pretrained import get_pretrained
        self.use_cuda = True
        self.pool = pool
        self._size = 100
        self._max_len = 800
        self.precomputed = False

        self._embed = lm_embed
        self._dscript_model = get_pretrained("human_v1")
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_BeplerBerger_DSCRIPT_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq, use_cuda=self.use_cuda)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            if self.pool:
                return ds_emb.squeeze().mean(axis=0)
            else:
                return ds_emb.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

class ESM_DSCRIPT_f:
    def __init__(self, pool=True, model_path=f"{MODEL_DIR}/esm_epoch5_state_dict.pt"):
        from dscript.models.embedding import FullyConnectedEmbed, SkipLSTM
        from dscript.models.contact import ContactCNN
        from dscript.models.interaction import ModelInteraction
        def build_human_esm(state_dict_path):
            """
            :meta private:
            """
            embModel = FullyConnectedEmbed(1280, 100, 0.5)
            conModel = ContactCNN(100, 50, 7)
            model = ModelInteraction(embModel, conModel, use_cuda=True, do_w=True, pool_size=9)
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        self.use_cuda = True
        self.pool = pool
        self._size = 100
        self._max_len = 800
        self.precomputed = False

        self._embed = ESM_f(pool=False)
        self._dscript_model = build_human_esm(model_path)
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript-esm protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ESM_DSCRIPT_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            if self.pool:
                return ds_emb.squeeze().mean(axis=0)
            else:
                return ds_emb.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

class ProtBert_DSCRIPT_f:
    def __init__(self, pool=True, model_path=f"{MODEL_DIR}/protbert_epoch3_state_dict.pt"):
        from dscript.models.embedding import FullyConnectedEmbed, SkipLSTM
        from dscript.models.contact import ContactCNN
        from dscript.models.interaction import ModelInteraction
        def build_human_protbert(state_dict_path):
            """
            :meta private:
            """
            embModel = FullyConnectedEmbed(1024, 100, 0.5)
            conModel = ContactCNN(100, 50, 7)
            model = ModelInteraction(embModel, conModel, use_cuda=True, do_w=True, pool_size=9)
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        self.use_cuda = True
        self.pool = pool
        self._size = 100
        self._max_len = 800
        self.precomputed = False

        self._embed = ProtBert_f(pool=False)
        self._dscript_model = build_human_protbert(model_path)
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript-protbert protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ProtBert_DSCRIPT_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            if self.pool:
                return ds_emb.squeeze().mean(axis=0)
            else:
                return ds_emb.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

##################################
# PLM + D-SCRIPT 100 Featurizers #
##################################

class BeplerBerger_DSCRIPT_cat_f:
    def __init__(self, pool=True):
        from dscript.language_model import lm_embed
        from dscript.pretrained import get_pretrained
        self.use_cuda = True
        self.pool = pool
        self._size = 6265
        self._max_len = 800
        self.precomputed = False

        self._embed = lm_embed
        self._dscript_model = get_pretrained("human_v1")
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_BeplerBerger_DSCRIPT_cat_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq,use_cuda=self.use_cuda)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            emb_cat = torch.cat((lm_emb,ds_emb),dim=2).squeeze()

            if self.pool:
                return emb_cat.squeeze().mean(axis=0)
            else:
                return emb_cat.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

class ESM_DSCRIPT_cat_f:
    def __init__(self, pool=True, model_path=f"{MODEL_DIR}/esm_epoch5_state_dict.pt"):
        from dscript.models.embedding import FullyConnectedEmbed, SkipLSTM
        from dscript.models.contact import ContactCNN
        from dscript.models.interaction import ModelInteraction
        def build_human_esm(state_dict_path):
            """
            :meta private:
            """
            embModel = FullyConnectedEmbed(1280, 100, 0.5)
            conModel = ContactCNN(100, 50, 7)
            model = ModelInteraction(embModel, conModel, use_cuda=True, do_w=True, pool_size=9)
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        self.use_cuda = True
        self.pool = pool
        self._size = 1380
        self._max_len = 800
        self.precomputed = False

        self._embed = ESM_f(pool=False)
        self._dscript_model = build_human_esm(model_path)
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript-esm protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ESM_DSCRIPT_cat_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            emb_cat = torch.cat((lm_emb,ds_emb),dim=1)

            if self.pool:
                return emb_cat.squeeze().mean(axis=0)
            else:
                return emb_cat.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)

class ProtBert_DSCRIPT_cat_f:
    def __init__(self, pool=True, model_path=f"{MODEL_DIR}/protbert_epoch3_state_dict.pt"):
        from dscript.models.embedding import FullyConnectedEmbed, SkipLSTM
        from dscript.models.contact import ContactCNN
        from dscript.models.interaction import ModelInteraction
        def build_human_protbert(state_dict_path):
            """
            :meta private:
            """
            embModel = FullyConnectedEmbed(1024, 100, 0.5)
            conModel = ContactCNN(100, 50, 7)
            model = ModelInteraction(embModel, conModel, use_cuda=True, do_w=True, pool_size=9)
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        self.use_cuda = True
        self.pool = pool
        self._size = 1124
        self._max_len = 800
        self.precomputed = False

        self._embed = ProtBert_f(pool=False)
        self._dscript_model = build_human_protbert(model_path)
        self._dscript_model.use_cuda = self.use_cuda
        if self.use_cuda:
            self._dscript_model = self._dscript_model.cuda()

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        print("--- precomputing dscript-protbert protein featurizer ---")
        assert not self.precomputed
        precompute_path = f"{to_disk_path}_ProtBert_DSCRIPT_cat_f_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
        if from_disk and os.path.exists(precompute_path):
            print("--- loading from disk ---")
            self.prot_embs = pk.load(open(precompute_path,"rb"))
        else:
            self.prot_embs = {}
            for sq in tqdm(seqs):
                if sq in self.prot_embs:
                    continue
                self.prot_embs[sq] = self._transform(sq)

            if to_disk_path is not None and not os.path.exists(precompute_path):
                print(f'--- saving protein embeddings to {precompute_path} ---')
                pk.dump(self.prot_embs, open(precompute_path,"wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[:self._max_len]

        with torch.no_grad():
            lm_emb = self._embed(seq)
            if self.use_cuda:
                lm_emb = lm_emb.cuda()
            ds_emb = self._dscript_model.embedding(lm_emb)
            emb_cat = torch.cat((lm_emb,ds_emb),dim=1)

            if self.pool:
                return emb_cat.squeeze().mean(axis=0)
            else:
                return emb_cat.squeeze()

    def __call__(self, seq):
        if self.precomputed:
            return self.prot_embs[seq]
        else:
            return self._transform(seq)
