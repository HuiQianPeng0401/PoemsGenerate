import torch
from torch.utils.data import Dataset
from torch import device


class PoetryData(Dataset):
    def __init__(
            self,
            device,
            *,
            token_length=12,
            poetry_file="./data/chinese_poems.txt",
            max_lines=100000,
    ):
        super(PoetryData,self).__init__()
        self.corpus=[]
        self.token_length=token_length
        self.idx2word=["<bos>","<eos>","<pad>"]
        self.word2idx={v : k for k, v in enumerate(self.idx2word)}
        idx=len(self.idx2word)
        self.device=device
        loaded_lines=0
        with open(poetry_file,"r",encoding="utf-8") as file:
            while loaded_lines < max_lines:
                line=file.readline().strip("\n\r")
                if len(line)==0:
                    break
                loaded_lines+=1
                for k in line:
                    if k not in self.word2idx:
                        self.word2idx[k]=idx
                        self.idx2word.append(k)
                        idx+=1
                for pair in line.split("."):
                    t=pair.split(",")
                    if len(t)==2:
                        if len(t[0])>6:
                            self.corpus.append((t[0],t[1]))
        self.vocab_size=len(self.word2idx)

    def word2token(self,words):
        t=[0]
        t.extend(self.word2idx[x] for x in words[:self.token_length-2])
        t.append(1)
        t.extend(2 for _ in range(max(0,self.token_length-len(t))))
        return torch.LongTensor(t).to(self.device)

    def token2word(self,tokens):
        return "".join(self.idx2word[x] for x in tokens)

    def get_token_mask(self,token):
        return (token==2).to(self.device)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        up, down=self.corpus[index]
        src=self.word2token(up)
        tgt=self.word2token(down)
        return (src,tgt)


