# PoemsGenerate
## 简介
用Transformer来生成诗歌，这个项目写了差不多有20多天，老实说效果并不是特别好，我对Transformer的理解还是太浅了，不知道怎样才能做得更好，学了一个多月的深度学习，感觉还是有一些问题没有搞清楚，先准备复试，希望能上岸，之后会继续学习，完善这个项目。\
部分训练结果如下\
![屏幕截图 2024-03-03 093550](https://github.com/HuiQianPeng0401/PoemsGenerate/assets/57929041/af37f601-a62d-400b-bb1d-95c046db9de8)
![屏幕截图 2024-03-03 094626](https://github.com/HuiQianPeng0401/PoemsGenerate/assets/57929041/c60d027a-2ff2-45b2-8ec0-e57ef67c0850)
## 数据准备
古诗数据的txt文件，[下载地址](https://www.kaggle.com/datasets/qianboao/chinesepoetrydataset)
## 数据处理
### 基本概念
>word2idx：每个字和它对应的序号，例如”春“这个字对应的序号是1000\
>idx2word：每个序号和它对应的字，例如序号1000对应着“春”这个字\
>BOS=0代表诗句开始，EOS=1代表结束，为了使长度一致，PAD=2代表填充
### 数据处理类
定义了一个名为PoetryData的类，继承自Dataset类。在初始化函数__init__中，进行了一些初始化操作。接下来进行了一些初始化工作，包括定义数据集、词汇表和标记索引等。
### 读取数据
在这部分代码中，打开指定的诗歌文件，并逐行读取数据。通过循环迭代每一行数据，进行以下操作：
使用strip("\n\r")函数删除行末尾的换行符。
如果当前行为空行，则终止循环。
遍历当前行的每个字符k，检查是否已经存在于word2idx中，如果不存在，则将其添加到word2idx中，并更新idx2word和idx。
使用line.split(".")将当前行按句号"."切分成多个句子。
对于每个切分得到的句子对pair，使用pair.split(",")按逗号","将其分割成上下句，并进行长度判断。
我只想保留七言律，诗如果上句的长度大于6，则将上下句的组合(t[0], t[1])添加到corpus列表中。
```
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
```
### 文字转换
函数word2token用于将词语序列转换为令牌序列，它接受一个词语列表words作为输入，并返回一个LongTensor对象。
函数内部首先创建一个空列表t，并在列表开头添加起始标记"<bos>"的索引0。然后使用生成器表达式self.word2idx[x] for x in words[:self.token_length-2]将词语序列转换为对应的索引序列，并通过extend方法将其添加到列表t中。
接下来将终止标记"<eos>"的索引1添加到列表末尾，并通过extend方法将长度不足的部分使用填充标记"<pad>"的索引2进行填充，直到达到token_length的长度要求。
最后返回转换后的令牌序列，并使用.to(self.device)将其转移到指定的设备上。
同样，函数token2word，用于将令牌序列转换回词语序列。它接受一个令牌序列tokens作为输入，并返回一个字符串。
函数内部使用生成器表达式self.idx2word[x] for x in tokens将令牌序列转换为对应的词语序列，并使用"".join()方法将它们连接成一个字符串。
```
def word2token(self, words):
        t = [0]
        t.extend(self.word2idx[x] for x in words[:self.token_length-2])
        t.append(1)
        t.extend(2 for _ in range(max(0, self.token_length-len(t))))
        return torch.LongTensor(t).to(self.device)
def token2word(self, tokens):
        return "".join(self.idx2word[x] for x in tokens)
```
### 获取诗句
重写getitem根据索引index获取数据集中的一个样本。它返回一个元组(src, tgt)，其中src是上句转换为的令牌序列，tgt是下句转换为的令牌序列。
函数内部根据索引index从corpus列表中获取对应的上句和下句，并分别通过调用word2token方法将它们转换为令牌序列。然后，将转换后的令牌序列作为元组返回。
```
def __getitem__(self, index):
        up, down = self.corpus[index]
        src = self.word2token(up)
        tgt = self.word2token(down)
        return (src, tgt)
```
## 模型定义
### 类的定义
PoetryModel的PyTorch模型类用于处理诗歌生成任务。\
vocab_size表示词汇表的大小，device表示模型所在的设备（如CPU或GPU）。\
embed_size表示嵌入层的维度，默认为512。\
n_head表示Transformer中注意力头的数量，默认为8。\
n_layer表示Transformer中编码器和解码器的层数，默认为4。\
hidden_size表示Transformer中前馈神经网络的隐藏层维度，默认为512。\
self.embed是一个嵌入层，用于将词汇表中的词语转换为密集向量表示。\
self.embed_size保存嵌入层的维度。\
self.sq保存嵌入层维度的平方根，用于缩放嵌入向量。\
self.transformer是一个Transformer模型，使用了nn.Transformer类。\
self.generator是一个线性层，用于将Transformer的输出映射回词汇表的维度。\
self.pos_encoding是一个位置编码器，用于为输入序列添加位置信息。位置编码器对训练非常重要，但是它的实现似乎比较固定，老实说为什么这样写我也不太明白。
```
class PoetryModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            device,
            *,
            embed_size=512,
            n_head=8,
            n_layer=4,
            hidden_size=512
    ):
        super(PoetryModel,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size,device=device)
        self.embed_size=embed_size
        self.sq=math.sqrt(self.embed_size)
        self.transformer=nn.Transformer(
            embed_size,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
            device=device
        )
        self.generator=nn.Linear(embed_size,vocab_size)
        self.pos_encoding=PositionalEncoding(embed_size)
```
### 前向传播
forward方法定义了模型的前向传播过程。\
src表示源序列的输入，tgt表示目标序列的输入。\
tgt_mask是用于掩盖解码器未来信息的掩码。\
src_padding_mask和tgt_padding_mask是用于掩盖输入序列中填充的掩码。\
self.embed将源序列和目标序列映射为嵌入向量，并乘以缩放因子。\
self.pos_encoding为输入序列添加位置编码。\
self.transformer进行Transformer模型的前向传播，将源序列和目标序列作为输入。\
out保存Transformer模型的输出。\
self.generator将Transformer的输出映射回词汇表的维度。\
最后返回out作为模型的输出。
```
def forward(
            self,
            src,
            tgt,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
    ):
        src=self.embed(src)*self.sq
        src=self.pos_encoding(src)
        tgt=self.embed(tgt)*self.sq
        tgt=self.pos_encoding(tgt)
        out=self.transformer.forward(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        out=self.generator.forward(out)
        return out
```
### 编码与解码
encode函数用于对源序列进行编码。它的输入是源序列数据src，以及可选的噪声参数noise和noise_intensity。在函数内部，首先通过嵌入层self.embed将源序列src转换为嵌入向量embeded。如果noise为True，将随机生成的噪声加到嵌入向量中，增加模型的鲁棒性。最后调用self.transformer.encoder.forward对嵌入向量进行编码，使用了位置编码器和缩放因子。编码后的结果作为encode函数的输出。
encode函数的主要作用是将源序列编码为一个表示，该表示将用于后续的解码过程。在模型训练过程中，编码器将源序列的信息编码为一个固定长度的向量，然后传递给解码器进行生成。因此，encode函数在训练过程中是非常重要的。
decode函数用于对目标序列进行解码。它的输入是目标序列数据tgt和编码后的源序列数据memory。在函数内部首先通过嵌入层self.embed将目标序列tgt转换为嵌入向量，然后通过位置编码器和缩放因子对嵌入向量进行处理。接下来调用self.transformer.decoder.forward对嵌入向量进行解码，使用了位置编码器和缩放因子。解码后的结果是一个表示生成序列的概率分布。最后，通过线性层self.generator.forward将概率分布映射回词汇表的维度，得到最终的生成结果。
decode函数的作用是根据编码后的源序列和目标序列生成诗句。在生成诗句的过程中，首先将目标序列转换为嵌入向量，并经过解码器进行解码，得到生成序列的概率分布。然后，通过线性层进行映射，得到生成序列的输出。在模型训练过程中，decode函数通常用于生成诗句的推断过程，即给定一个初始的词语或句子作为输入，逐步生成后续的词语，直到生成结束。
>需要注意的是，在训练过程中，encode函数和decode函数并不直接用于模型的训练。它们是模型的组成部分，用于在训练完成后生成诗句。在训练过程中，通过调用forward函数来完成模型的前向传播和参数更新。而在生成诗句时，可以利用经过训练优化参数的self.transformer，调用encode函数和decode函数来生成诗句。
```
def encode(
            self,
            src,
            noise=True,
            noise_intensity=1,
    ):
        embeded=self.embed.forward(src)
        if noise:
            embeded+=torch.rand_like(embeded)*noise_intensity
        return self.transformer.encoder.forward(
            self.pos_encoding.forward(embeded*self.sq)
        )
def decode(
            self,
            tgt,
            memory,
    ):
        return self.generator.forward(
            self.transformer.decoder.forward(
                self.pos_encoding.forward(self.embed(tgt)*self.sq),
                memory,
            )
        )
```
## 训练
### 类的定义
创建一个Poetry类来辅助训练，需要注意几个变量的定义。\
PoetryData对象，将其赋值给self.dataset。PoetryData是上面的数据集类，用于加载和处理诗歌数据。这里传递了设备类型、最大行数和令牌长度作为参数。\
将self.dataset的词汇表大小赋值给self.vocab_size，以便后续模型的构建和训练。\
使用random_split函数将数据集self.dataset划分为训练数据和测试数据，并分别赋值给train_data和test_data。训练数据集的大小是总数据集大小减去1000，测试数据集大小为1000。\
PoetryModel对象，将其赋值给self.model。PoetryModel是一个自定义的模型类，用于生成诗句。\
Adam优化器对象self.optimizer，用于优化模型的参数。self.model.parameters()返回模型中需要被优化的参数列表。\
一个学习率调度器self.optimizer_scheduler，使用余弦退火调整优化器的学习率。self.optimizer是被调度的优化器对象，256是调度器的周期。\
交叉熵损失函数对象self.loss，用于计算模型的损失值。在前面的数据处理中，填充项PAD的值为2，所以需要让ignore_index=2来在计算损失时忽略标签为2的项。
### 训练
定义一个训练一个epoch的方法train_epoch()，其中需要注意的是在得到预测输出后计算loss时，loss=loss_f.forward(out.reshape(-1,vocab_size),tgt[:,1:].flatten())计算损失值。这里要将输出out进行形状重塑，然后与目标序列的子序列（去掉起始标记）进行比较，使用损失函数计算损失值。\
定义一个辅助方法forward_model()，用于进行模型的前向传播，这个方法是训练的核心。
src_mask=(src==2).to(self.device)：创建源序列的掩码，标记为2的位置为True，其余位置为False。这里的掩码用于指示模型在进行自注意力计算时忽略掉标记为2的位置。\
dec_tgt=tgt[:,:-1]：截取目标序列的子序列，去掉最后一个标记。这是因为在训练过程中，模型需要根据前面的标记预测下一个标记，因此最后一个标记不参与训练。\
dec_tgt_mask=(dec_tgt==2).to(self.device)：创建目标序列的掩码，同样将标记为2的位置设为True。\
tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(dec_tgt.size(1)).to(self.device)：生成目标序列的遮罩，用于在解码器中屏蔽未来标记。因为是自回归解码器，所以这里使用了generate_square_subsequent_mask()函数生成一个下三角矩阵，确保在解码时只能看到当前位置和之前的位置，而不能看到未来的位置。\
out=self.model.forward(src,dec_tgt,tgt_mask,src_mask,dec_tgt_mask)：调用模型的forward()方法进行前向传播，传入源序列、解码器输入、目标序列遮罩、源序列掩码和目标序列掩码。返回的out是模型的输出。
```
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
```
## 诗句生成
通过word2idx将起始词列表转换为对应的索引，并创建一个目标序列(tgt)，其中包含起始标记。
使用word2token()方法将预设的句子转换为标记序列(src)。这个句子序列会被送入编码器(encode())中。
编码器的encode()方法会接收标记序列(src)作为输入，并返回编码器的输出(memo)。这些输出将作为解码器的记忆，用于生成诗句的每个标记。
创建一个空列表(res)，用于存储生成的诗句的标记。
开始生成诗句的循环，循环次数为12次（在代码中规定的）。
在每次循环中，调用解码器的decode()方法，并传入目标序列(tgt)和记忆(memo)作为输入。解码器将根据这些输入生成下一个标记的输出(out)。
使用argmax(2)方法找到输出(out)中概率最高的词的索引，即预测生成的下一个标记。
如果生成的下一个标记是结束标记EOS，则跳出循环，表示诗句生成结束。
将生成的下一个标记添加到结果列表(res)中。
将生成的下一个标记拼接到目标序列(tgt)的末尾，以便在下一次迭代中生成下一个标记。
循环结束后，将起始词和生成的标记序列转换为诗句，并返回生成的诗句。
```
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
            print(next_word.size())
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
```
