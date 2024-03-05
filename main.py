import torch
import time
from torch.utils.data import DataLoader,random_split
from data import PoetryData
from model import PoetryModel
from tqdm import tqdm


batch_size=64
lr=0.0001


class Poetry:
    def __init__(self):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = torch.device("cuda:0")
        self.dataset=PoetryData(self.device,max_lines=50000,token_length=12)
        self.vocab_size=self.dataset.vocab_size
        train_data,test_data=random_split(self.dataset,[len(self.dataset)-1000,1000])
        self.train_dataloader=DataLoader(train_data,batch_size,True)
        self.test_dataloader=DataLoader(test_data,batch_size,True)
        self.model=PoetryModel(self.vocab_size,self.device,embed_size=512).to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr)
        self.optimizer_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,256)
        self.loss=torch.nn.CrossEntropyLoss(ignore_index=2)
        self.loaded_checkpoint_file=None
        self.epoch=0
        import glob
        files=glob.glob("checkpoint-*.pth")
        for i,file in enumerate(files):
            print(f"{i}->{file}")
        if files:
            t=input("输入序号来选择checkpoint，默认最后一个，输入'n'不选择>")
            if t=="":
                t=-1
            if t!="n":
                self.load_checkpoint(files[int(t)])


    def save_checkpoint(self):
        file_name = (
            self.loaded_checkpoint_file
            or f'checkpoint-{time.strftime("%y%m%d-%H%M")}.pth'
        )
        with open(file_name,"wb") as file:
            torch.save(
                {
                    "net_state":self.model.state_dict(),
                    "optimizer_state":self.optimizer.state_dict(),
                    "epoch":self.epoch
                },
                file,
            )
        print(f"save checkpoint to {file_name}")
        self.loaded_checkpoint_file=file_name

    def load_checkpoint(self,file):
        ckpt=torch.load(file)
        self.model.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch=ckpt["epoch"]
        self.loaded_checkpoint_file=file
        self.optimizer_scheduler.last_epoch=self.epoch
        print(f"loaded checkpoint: {file}, epoch: {self.epoch}")


    def generate_one(self,pre_sentence,start_words=""):
        self.model.eval()
        start_words_token=[0]
        start_words_token.extend(self.dataset.word2idx[x] for x in start_words)
        #src.shape=(N,),src->unsqueeze(0)->src.shape=(1,N)
        src=self.dataset.word2token(pre_sentence).unsqueeze(0)
        tgt=torch.LongTensor([start_words_token]).to(self.device)
        memo=self.model.encode(src)
        res=[]
        for i in range(12):
            out=self.model.decode(tgt,memo)
            #找到概率最高的词的索引
            next_word=out.argmax(2)
            if next_word[0][-1]==1: # <eos>说明结尾，该退出了
                break
            res.append(next_word[0][-1].item())
            tgt=torch.cat((tgt,next_word[:,-1:]),1)
        return start_words+self.dataset.token2word(res)

    def generate(self,num_sentence,pre_style):
        res=[]
        for i in range(num_sentence):
            s=self.generate_one(pre_style if not res else res[-1])
            res.append(s)
        return "/".join(res)

    def generate_by_start(self,start_words,pre_style):
        res=[]
        start_words_list=start_words.split("/")
        if not start_words_list:
            return ""
        for i,s in enumerate(start_words_list):
            t=self.generate_one(pre_style if not res else res[-1],s)
            res.append(t)
        return "/".join(res)

    def forward_model(self,src,tgt):
        # src.shape,tgt.shape = torch.Size([batch_size, 12]) = torch.Size([batch_size, token_length])
        src=src.to(self.device)
        tgt=tgt.to(self.device)
        src_mask=(src==2).to(self.device)
        # dec_tgt.shape = torch.Size([batch_size, 11]) = torch.Size([batch_size, token_length - 1])
        dec_tgt=tgt[:,:-1]
        dec_tgt_mask=(dec_tgt==2).to(self.device)
        tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(dec_tgt.size(1)).to(self.device)
        out=self.model.forward(src,dec_tgt,tgt_mask,src_mask,dec_tgt_mask)
        return out

    def train_epoch(self):
        self.model.train()
        loss_f=self.loss
        vocab_size=self.dataset.vocab_size
        len_data=len(self.train_dataloader)
        loss_sum=0
        for i,(src,tgt) in tqdm(enumerate(self.train_dataloader)):
            #src.shape,tgt.shape = torch.Size([batch_size, 12]) = torch.Size([batch_size, token_length])
            out=self.forward_model(src,tgt)

            #out.shape = torch.Size([batch_size, 11, vocab_size])

            loss=loss_f.forward(out.reshape(-1,vocab_size),tgt[:,1:].flatten())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum+=loss.item()
        print(loss_sum)
        self.optimizer_scheduler.step()


    def evalution(self):
        self.model.eval()
        loss=self.loss
        vocab_size=self.dataset.vocab_size
        loss_sum=0
        with torch.no_grad():
            for i,(src,tgt) in enumerate(self.test_dataloader):
                out=self.forward_model(src,tgt)
                l=loss.forward(out.reshape(-1,vocab_size),tgt[:,1:].flatten())
                loss_sum+=l.item()



    def training(self,epoch_num=10):
        for i in range(epoch_num):
            self.train_epoch()
            self.evalution()
            self.epoch+=1
            self.save_checkpoint()
            print(self.generate(4,"日照香炉生紫烟"))


def main():
    model = Poetry()
    model.training(200)
    """
    # print(model.generate())
    while True:
        s = input(">")
        if s == "exit":
            break
        if s.find("/") != -1:
            print(model.generate_by_start(s, "落魄江湖载酒行"))
        else:
            print(model.generate(4, s or "床前明月光"))
    """


if __name__ == "__main__":
     main()