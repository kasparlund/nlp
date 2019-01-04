from fastai.text import * 
def usedGB_RAM(): 
    import psutil
    return round((psutil.virtual_memory().used + psutil.swap_memory().used)/1e9,2)
class BatchLayout(Enum):
    Parallel   = 1
    Sequential = 2
class MyLanguageModelLoader():
    "Create a dataloader with bptt slightly changing."
    
    class CircularIndex():
        #When the index exceeds the length of self.idx then it is wrap to start at the head 
        #or the end if the indexing i set to move backwards
        def __init__(self, length:int, forward:bool): 
            self.idx      = np.arange(length)
            self.forward_ = forward
        def __getitem__(self, i): 
            index = i%len(self.idx) if self.forward_ else len(self.idx)-1-i%len(self.idx)
            return self.idx[index]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)
        def forward(self, forward:bool=True): self.forward_ = forward
                
    def __init__(self, dataset:LabelList, bs:int=64, bptt:int=70, backwards:bool=False, shuffle:bool=False,
                 max_len:int=25, p_bptt:int=0.95, bl:BatchLayout=BatchLayout.Sequential):
        self.init_kwargs = dict(bs=bs, bptt=bptt, backwards=backwards, shuffle=shuffle, max_len=max_len)
        self.dataset,self.bs,self.bptt,self.backwards,self.shuffle,self.p_bptt = dataset,bs,bptt,backwards,shuffle,p_bptt
        
        nToks = 0
        for s in dataset.x.items: nToks+=len(s)
        self.ite_len = math.ceil( nToks / (self.bs*self.bptt) ) #this is returned in def __len__(self) 

        self.idx = None
        self.buffer = None

        #self.min_seq,self.max_seq = 5,max_len #self.min_seq, self.max_seq is no longer used
        self.num_workers = 0
        self.bl = bl

        #self.minToks = 4 #argument used to discard end of sections that are too short to be used for prediction of the next word
        print(f"LanguageModelLoader.__init__ Used GB memory:{usedGB_RAM()} batches:{len(self)} nToks:{nToks} "+\
              f"bptt:{self.bptt} p_bptt:{self.p_bptt} shuffle:{self.shuffle} backwards:{self.backwards}" )  


    def allocate_buffers(self):     
        "allocate the required worth-case batchbuffer"

        #CircularIndex is used to index the raggged dataset. It 1) can shuffled in-place, 
        # 2)can be configured to move backward, 3) it will warp around when the index 
        # exceeds the length of the underlying ragged array
        self.idx = MyLanguageModelLoader.CircularIndex(len(self.dataset), not self.backwards)        

        #The batches vary uniformly around bppt as defined by p_bptt
        max_batch_element = self.bs * (1 + math.ceil( self.bptt*(1+0.5*self.p_bptt) ) )
        self.buffer       = np.zeros(max_batch_element, dtype=np.long)
        
        if self.bl is BatchLayout.Parallel:self.ri = np.zeros((bs,2), dtype=np.int) #with col1=current rag and col2=index into the current rag
        print(f"LanguageModelLoader.allocate_buffers Used GB memory:{usedGB_RAM()} "+\
              f"shuffle:{self.shuffle} backwards:{self.backwards}" )  
        
    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None: 
            yield LongTensor(getattr(self.dataset, 'item'))[None],LongTensor([0])

        #allocate buffers lazily in order to avoid vasting memory on fix_ds which is not always used/may never of ULMFit
        if self.idx is None: self.allocate_buffers()
        if self.shuffle: self.idx.shuffle()

        if self.bl is BatchLayout.Parallel:
            step         = len(self.idx)//bs #step is trucated => batches may overlap a bit
            self.ri[:,0] = np.arange(self.bs)*step
            self.ri[:,1] = 0
        else:
            self.ei,self.eo = 0,0

        i = 0
        while i < self.ite_len:
            seq_len = int(self.bptt*(1. + self.p_bptt*(np.random.random() - 0.5)))
            if self.bl is BatchLayout.Parallel:
                self.parallel_fill_buffer(seq_len+1)
                data = self.buffer.reshape(self.bs,-1)
            else:    
                nToks   = self.bs*(seq_len+1)
                self.fill_buffer(nToks)
                data = self.buffer[:nToks].reshape(self.bs,-1)

            data  = torch.as_tensor( data, dtype=torch.long )
            res   = data[:,0:seq_len-1], data[:,1:seq_len]        
            i    += 1
            #if i==self.ite_len : print(res) # check that y is shift to predict x       
            yield res        
    
    def fill_buffer(self, nToks:int):
        "new the tokens in the buffer with nToks from the ragged array"
        #nToks: number of tokens to be extract and inserted starting at the beginning of the buffer from the 
        #       last saved indices in the ragged array and forward - possibly wrapping to the head of the dataset
        #ei: index of the last rag to be extract
        #eo: index (not inclusive) where the extract stops in the last rag
        #ei and eo starts at 0,0 in the beginning of a batch and is then updated as we fill in the buffer 
        #in the batch and batch by batch
        
        j, ibuf = self.ei, 0
        ei,eo   = self.ei, self.eo  #local variables are twice as fast in a loop
        #bi, bo  = self.ei, self.eo  #only used in the print statement below
        
        while nToks > ibuf:   
            rag = self.dataset.x.items[self.idx[j]]
            r0  = eo if j==ei else 0         
            r1  = len(rag)
            rl  = r1 - r0
            if ibuf+rl >= nToks:
                eo = (nToks + eo) if j==ei else (nToks-ibuf) 
                ei = j
                r1 = eo
                rl = r1 - r0
            self.buffer[ibuf:ibuf+rl] = rag[r0:r1]
            ibuf += rl
            j    += 1      
        self.ei, self.eo = ei, eo 
        #if self.ei==bi: print( f"one rag:{self.ei==bi} nToks:{nToks} nBToks:{ibuf} bi:{bi} bo:{bo} ei:{self.ei} eo:{self.eo}" )        

    def parallel_fill_buffer(self,  seq_len:int):
        "new the tokens in the buffer with nToks from the ragged array"
        #nToks: number of tokens to be extract and inserted starting at the beginning of the buffer from the 
        #       last saved indices in the ragged array and forward - possibly wrapping to the head of the dataset
        #ei: index of the last rag to be extract
        #eo: index (not inclusive) where the extract stops in the last rag
        #ei and eo starts at 0,0 in the beginning of a batch and is then updated as we fill in the buffer 
        #in the batch and batch by batch
        
        j, ibuf = self.ri[:,0], self.ri[:,0]*0
        ei,eo   = self.ri[:,0], self.ri[:,1]
        bv      = self.buffer.reshape(len(self.ri),-1)
        fill    = (seq_len <= ibuf)
        #bi, bo = np.copy(ei), np.copy(eo)  #only used in the print statement below
        while fill.any():   
            rags = np.asarray( self.dataset.x.items[self.idx[j]] )
            r0   = np.where(j==ei, eo, 0)
            r1   = np.asarray( [len(r) for r in rags] )
            rl   = r1 - r0
            last = ibuf+rl >= seq_len
            if last.any():
                eo[last] = np.where( j==ei, (seq_len + eo), (seq_len-ibuf) )[last]
                ei[last] = j[last]
                r1[last] = eo[last]
                rl[last] = (r1 - r0)[last]
            bv[fill, ibuf[fill]:ibuf[fill]+rl[fill]] = rags[fill,r0[fill]:r1[fill]]
            ibuf[fill] += rl[fill]
            j[fill]    += 1      
            fill        = (seq_len <= ibuf)

        self.ri[:,0] = ei
        self.ri[:,1] = eo
        #if self.ei==bi: print( f"one rag:{self.ei==bi} seq_len:{seq_len} ibuf:{ibuf} bi:{bi} bo:{bo} ei:{self.ei} eo:{self.eo}" )        

    def __len__(self) -> int: return self.ite_len
    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

    @property
    def batch_size(self): return self.bs
    @batch_size.setter
    def batch_size(self, v): self.bs = v

    def batchify(self, data:np.ndarray) -> LongTensor: pass

class MyTextLMDataBunch(TextLMDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, 
                 train_ids:Collection[Collection[int]],        valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, 
                 train_lbls:Collection[Union[int,float]]=None, valid_lbls:Collection[Union[int,float]]=None, 
                 classes:Collection[Any]=None, processor:PreProcessor=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a `vocab`."
        src = LabelLists(path, TextList(train_ids, vocab, path=path, processor=[]),
                               TextList(valid_ids, vocab, path=path, processor=[]))
        #src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_lists(train_lbls, valid_lbls, classes=classes, processor=[]) 
        src.train = src.train.label_for_lm()
        src.valid = src.valid.label_for_lm()

        #if test_ids is not None: src.add_test(TextList(test_ids, vocab, path=path), label=train_lbls[0])
        #src.valid.x.processor = ifnone(processor, [TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
       
        #ensure our create is called
        src.train.x._bunch = MyTextLMDataBunch
        src.valid.x._bunch = MyTextLMDataBunch
        return src.databunch(**kwargs)
    
    #need customized version of this in order to set MyLanguageModelLoader
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        print("MyTextLMDataBunch def create")
        datasets    = cls._init_ds(train_ds, valid_ds, test_ds)
        dataloaders = [MyLanguageModelLoader(ds, shuffle=(i==0), **kwargs) for i,ds in enumerate(datasets)]
        return cls(*dataloaders, path=path, no_check=no_check)
