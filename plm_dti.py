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

PRECOMPUTED_PROTEIN_PATH = "precomputed_proteins.pk"
PRECOMPUTED_MOLECULE_PATH = "precomputed_molecules.pk"

class Random_f:
    def __init__(self, pool=True):
        self.use_cuda = True

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        pass

    def _transform(self, seq):
        return torch.rand(100).cuda()

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

class ESM_f:
    def __init__(self,
                 pool: bool = True,
                 dl_path: str = "/afs/csail/u/s/samsl/Work/TorchHub",
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

        self._protbert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self._protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert")
        if self.use_cuda:
            self._protbert_feat = pipeline('feature-extraction', model=self._protbert_model, tokenizer=self._protbert_tokenizer,device=0)
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
    def __init__(self, pool=True, model_path='/afs/csail/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/ESM_dscript/esm_epoch5_state_dict.pt'):
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
    def __init__(self, pool=True, model_path='/afs/csail/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/ESM_dscript/protbert_epoch3_state_dict.pt'):
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
    def __init__(self, pool=True, model_path='/afs/csail/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/ESM_dscript/esm_epoch5_state_dict.pt'):
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
    def __init__(self, pool=True, model_path='/afs/csail/u/s/samsl/Work/DSCRIPT_Dev_and_Testing/ESM_dscript/protbert_epoch3_state_dict.pt'):
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

#####################################
# Protein Interface Representations #
#####################################

# class DPatch_f:
#     def __init__(self, pool=True):
#         from dscript.language_model import lm_embed
#         from dscript.pretrained import get_pretrained
#         self.use_cuda = True
#         self.pool = pool
#         self._size = 50
#         self._max_len = 800
#         self.precomputed = False

#         self._embed = lm_embed
#         self._dscript_model = get_pretrained("human_v1")
#         self._dscript_model.use_cuda = self.use_cuda
#         self._pbe = PB_Embed.load_from_checkpoint("/afs/csail.mit.edu/u/s/samsl/Work/Interface_Prediction/src/pb_embed_20210425.ckpt")
#         if self.use_cuda:
#             self._dscript_model = self._dscript_model.cuda()
#             self._pbe = self._pbe.cuda()

#     def precompute(self, seqs, to_disk_path=True, from_disk=True):
#         print("--- precomputing dpatch protein featurizer ---")
#         assert not self.precomputed
#         precompute_path = f"{to_disk_path}_DPATCH_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
#         if from_disk and os.path.exists(precompute_path):
#             print("--- loading from disk ---")
#             self.prot_embs = pk.load(open(precompute_path,"rb"))
#         else:
#             self.prot_embs = {}
#             for sq in tqdm(seqs):
#                 if sq in self.prot_embs:
#                     continue
#                 self.prot_embs[sq] = self._transform(sq)

#             if to_disk_path is not None and not os.path.exists(precompute_path):
#                 print(f'--- saving protein embeddings to {precompute_path} ---')
#                 pk.dump(self.prot_embs, open(precompute_path,"wb+"))
#         self.precomputed = True

#     @lru_cache(maxsize=5000)
#     def _transform(self, seq):
#         if len(seq) > self._max_len:
#             seq = seq[:self._max_len]

#         with torch.no_grad():
#             lm_emb = self._embed(seq, use_cuda=self.use_cuda)
#             if self.use_cuda:
#                 lm_emb = lm_emb.cuda()
#             ds_emb = self._dscript_model.embedding(lm_emb)
#             ds_emb = ds_emb.squeeze().mean(axis=0)
#             return self._pbe(ds_emb)

#     def __call__(self, seq):
#         if self.precomputed:
#             return self.prot_embs[seq]
#         else:
#             return self._transform(seq)

#########################
# Molecular Featurizers #
#########################

class Morgan_f:
    def __init__(self,
                 size=2048,
                 radius=2,
                ):
        import deepchem as dc
        self._dc_featurizer = dc.feat.CircularFingerprint()
        self._size = size
        self.use_cuda = True
        self.precomputed = False

    def precompute(self, smiles, to_disk_path=None, from_disk=True):
        print("--- precomputing morgan molecule featurizer ---")
        assert not self.precomputed

        if from_disk and os.path.exists(f"{to_disk_path}_Morgan_MOLECULES.pk"):
            print("--- loading from disk ---")
            self.mol_embs = pk.load(open(f"{to_disk_path}_Morgan_MOLECULES.pk","rb"))
        else:
            self.mol_embs = {}
            for sm in tqdm(smiles):
                if sm in self.mol_embs:
                    continue
                m_emb = self._transform(sm)
                if len(m_emb) != self._size:
                    m_emb = torch.zeros(self._size)
                    if self.use_cuda:
                        m_emb = m_emb.cuda()
                self.mol_embs[sm] = m_emb
            if to_disk_path is not None and not os.path.exists(f"{to_disk_path}_Morgan_MOLECULES.pk"):
                print(f'--- saving morgans to {f"{to_disk_path}_Morgan_MOLECULES.pk"} ---')
                pk.dump(self.mol_embs, open(f"{to_disk_path}_Morgan_MOLECULES.pk","wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, smile):
        tens = torch.from_numpy(self._dc_featurizer.featurize([smile])).squeeze().float()
        if self.use_cuda:
            tens = tens.cuda()

        return tens

    def __call__(self, smile):
        if self.precomputed:
            return self.mol_embs[smile]
        else:
            return self._transform(smile)

class Mol2Vec_f:
    def __init__(self,
                 radius=1,
                ):
        import deepchem as dc
        self._dc_featurizer = dc.feat.Mol2VecFingerprint()
        self._size = 300
        self.use_cuda = True
        self.precomputed = False

    def precompute(self, smiles, to_disk_path=None, from_disk=True):
        print("--- precomputing mol2vec molecule featurizer ---")
        assert not self.precomputed

        if from_disk and os.path.exists(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"):
            print("--- loading from disk ---")
            self.mol_embs = pk.load(open(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk","rb"))
        else:
            self.mol_embs = {}
            for sm in tqdm(smiles):
                if sm in self.mol_embs:
                    continue
                m_emb = self._transform(sm)
                if len(m_emb) != self._size:
                    m_emb = torch.zeros(self._size)
                    if self.use_cuda:
                        m_emb = m_emb.cuda()
                self.mol_embs[sm] = m_emb
            if to_disk_path is not None and not os.path.exists(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"):
                print(f'--- saving morgans to {f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"} ---')
                pk.dump(self.mol_embs, open(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk","wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, smile):
        tens = torch.from_numpy(self._dc_featurizer.featurize([smile])).squeeze().float()
        if self.use_cuda:
            tens = tens.cuda()

        return tens

    def __call__(self, smile):
        if self.precomputed:
            return self.mol_embs[smile]
        else:
            return self._transform(smile)

##################
# Data Set Utils #
##################

def molecule_protein_collate_fn(args, pad=False):
    """
    Collate function for PyTorch data loader.

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    memb = [a[0] for a in args]
    pemb = [a[1] for a in args]
    labs = [a[2] for a in args]

    if pad:
        proteins = pad_sequence(pemb,batch_first=True)
    else:
        proteins = torch.stack(pemb, 0)
    molecules = torch.stack(memb, 0)
    affinities = torch.stack(labs, 0)

    return molecules, proteins, affinities

class DTIDataset(Dataset):
    def __init__(self,smiles, sequences, labels, mfeats, pfeats):
        assert len(smiles) == len(sequences)
        assert len(sequences) == len(labels)
        self.smiles = smiles
        self.sequences = sequences
        self.labels = labels

        self.mfeats = mfeats
        self.pfeats = pfeats

    def __len__(self):
        return len(self.smiles)

    @property
    def shape(self):
        return self.mfeats._size, self.pfeats._size

    def __getitem__(self, i):
        memb = self.mfeats(self.smiles[i])
        pemb = self.pfeats(self.sequences[i])
        lab = torch.tensor(self.labels[i])

        return memb, pemb, lab

#######################
# Model Architectures #
#######################

class SimpleCosine(nn.Module):
    def __init__(self,
                 mol_emb_size = 2048,
                 prot_emb_size = 100,
                 latent_size = 1024,
                 latent_activation = nn.ReLU
                ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size),
            latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size),
            latent_activation()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)

class SimpleCosineLinear(nn.Module):
    def __init__(self,
                 mol_emb_size = 2048,
                 prot_emb_size = 100,
                 latent_size = 1024,
                 latent_activation = nn.ReLU
                ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size),
            latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size),
            latent_activation()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)

class LSTMCosine(nn.Module):
    def __init__(self,
                 mol_emb_size = 2048,
                 prot_emb_size = 100,
                 lstm_layers = 3,
                 lstm_dim = 256,
                 latent_size = 256,
                 latent_activation = nn.ReLU,
                ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size),
            latent_activation()
        )

        self.rnn = nn.LSTM(self.prot_emb_size, lstm_dim, num_layers = lstm_layers, batch_first=True, bidirectional=True)

        self.prot_projector = nn.Sequential(
            nn.Linear(2*lstm_layers*lstm_dim, latent_size),
            nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1,0,2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(self,
                 mol_emb_size = 2048,
                 prot_emb_size = 100,
                 latent_size = 1024,
                 hidden_size = 4096,
                 latent_activation = nn.ReLU
                ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size),
            latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)

class SimpleConcat(nn.Module):
    def __init__(self,
                 mol_emb_size = 2048,
                 prot_emb_size = 100,
                 hidden_dim_1 = 512,
                 hidden_dim_2 = 256,
                 activation = nn.ReLU
                ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1),
            activation()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            activation()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid()
        )

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb],axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()

#################
# API Functions #
#################

MOL_FEATURIZERS = {
    "Morgan_f": Morgan_f,
    "Mol2Vec_f": Mol2Vec_f,
}

PROT_FEATURIZERS = {
    "Random_f": Random_f,
    "BeplerBerger_f": BeplerBerger_f,
    "BeplerBerger_DSCRIPT_f": BeplerBerger_DSCRIPT_f,
    "BeplerBerger_DSCRIPT_cat_f": BeplerBerger_DSCRIPT_cat_f,
    "ESM_f": ESM_f,
    "ESM_DSCRIPT_f": ESM_DSCRIPT_f,
    "ESM_DSCRIPT_cat_f": ESM_DSCRIPT_cat_f,
    "ProtBert_f": ProtBert_f,
    "ProtBert_DSCRIPT_f": ProtBert_DSCRIPT_f,
    "ProtBert_DSCRIPT_cat_f": ProtBert_DSCRIPT_cat_f,
    "DPatch_f": DPatch_f,
}

def get_dataloaders(train_df,
                    val_df,
                    test_df,
                    batch_size,
                    shuffle,
                    num_workers,
                    mol_feat,
                    prot_feat,
                    pool = True,
                    precompute=True,
                    drop_last=True,
                    to_disk_path=None,
                    device=0,
                  ):

    df_values = {}
    all_smiles = []
    all_sequences = []
    for df, set_name in zip([train_df, val_df, test_df], ["train", "val", "df"]):
        all_smiles.extend(df["SMILES"])
        all_sequences.extend(df["Target Sequence"])
        df_thin = df[["SMILES","Target Sequence","Label"]]
        df_values[set_name] = (df["SMILES"], df["Target Sequence"], df["Label"])

    mol_feats = MOL_FEATURIZERS[mol_feat]()
    prot_feats = PROT_FEATURIZERS[prot_feat](pool=pool)
    if precompute:
        mol_feats.precompute(all_smiles,to_disk_path=to_disk_path,from_disk=True)
        prot_feats.precompute(all_sequences,to_disk_path=to_disk_path,from_disk=True)

    loaders = {}
    for set_name in ["train", "val", "df"]:
        smiles, sequences, labels = df_values[set_name]

        dataset = DTIDataset(smiles, sequences, labels, mol_feats, prot_feats)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool))
        loaders[set_name] = dataloader

    return tuple([*loaders.values(), mol_feats._size, prot_feats._size])

def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size":32,
        "num_workers":0,
        "precompute":True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        "latent_size": 1024,
    }
    training_cfg = {
        "n_epochs": 50,
        "every_n_val": 1,
    }
    cfg = {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "experiment_id": experiment_id
    }

    return OmegaConf.structured(cfg)

def get_model(model_type, **model_kwargs):
    if model_type == "SimpleCosine":
        return SimpleCosine(**model_kwargs)
    elif model_type == "SimpleConcat":
        return SimpleConcat(**model_kwargs)
    elif model_type == "DeepCosine":
        return DeepCosine(**model_kwargs)
    elif model_type == "LSTMCosine":
        return LSTMCosine(**model_kwargs)
    else:
        raise ValueError("Specified model is not supported")


