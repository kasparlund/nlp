from fastai.text import * 

class BatchLayout(IntEnum):
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
            idx = self.idx
            return idx[ i%len(idx) if self.forward_ else len(idx)-1-i%len(idx) ]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)
        def forward(self, forward:bool=True): self.forward_ = forward

    def __init__(self, dataset:LabelList, bs:int=64, bptt:int=70, backwards:bool=False, shuffle:bool=False,
                 max_len:int=25, p_bptt:int=0.95, bl:BatchLayout=BatchLayout.Parallel):
        self.init_kwargs = dict(bs=bs, bptt=bptt, backwards=backwards, shuffle=shuffle, max_len=max_len)
        self.dataset,self.bs,self.bptt,self.backwards,self.shuffle,self.p_bptt = dataset,bs,bptt,backwards,shuffle,p_bptt
        
        self.first = True
        nToks = 0
        for i,s in enumerate(dataset.x.items): nToks += len(s)
        self.totalToks = nToks    
        self.ite_len   = math.ceil( nToks / (self.bs*self.bptt) ) #this is returned in def __len__(self) 
        self.idx       = None
        self.buffer    = None
        self.num_workers = 0
        self.bl = bl

        print(f"LanguageModelLoader.__init__ iterations:{len(self)} rags:{len(self.dataset.x.items)} nToks:{nToks} "+\
              f"bptt:{self.bptt} p_bptt:{self.p_bptt} shuffle:{self.shuffle} backwards:{self.backwards}" )  
        
    def __len__(self) -> int: return self.ite_len

    def allocate_buffers(self):     
        "allocate the required worth-case batchbuffer"

        #CircularIndex is used to index the raggged dataset. It 1) can shuffled in-place, 
        # 2)can be configured to move backward, 3) it will warp around when the index 
        # exceeds the length of the underlying ragged array
        self.idx = MyLanguageModelLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)        

        #The batches vary uniformly around bppt as defined by p_bptt
        max_batch_element = int( self.bs * (1 + math.ceil( self.bptt*(1+0.5*self.p_bptt) ) ) )
        self.buffer       = np.zeros(max_batch_element, dtype=np.long)
        
        if self.bl == BatchLayout.Parallel:
            self.ei = np.zeros(self.bs, dtype=np.int)
            self.eo = np.zeros(self.bs, dtype=np.int)
        print(f"LanguageModelLoader.allocate_buffers shuffle:{self.shuffle} backwards:{self.backwards} self.ite_len:{self.ite_len}" )  
        
    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None: 
            yield LongTensor(getattr(self.dataset, 'item'))[None],LongTensor([0])

        #allocate buffers lazily in order to avoid vasting memory on fix_ds which is not always used/may never of ULMFit
        if self.idx is None: self.allocate_buffers()
        if self.shuffle:     self.idx.shuffle()

        if self.bl == BatchLayout.Parallel:
            #It runs thourgh the rags and set an offset where each row in the batch begins
            #It reproduces how sgugger composes the offset of batches. This could be done in a much simpler 
            # by deviating from sguggers approach
            
            #stepTokens = int(math.ceil( self.totalToks/self.bs)) 
            stepTokens = self.totalToks/self.bs 
            self.ei[0] = self.eo[0] = i_row = countTokens = 0
            for i in range( len( self.dataset.x.items) ):
                countTokens += len( self.dataset.x.items[self.idx[i]] )
                while countTokens > int( (i_row+1) * stepTokens) and i_row+1 < self.bs:
                    i_row         += 1
                    self.ei[i_row] = i
                    self.eo[i_row] = countTokens - int(i_row*stepTokens)

            #print out for testing 
            lns = np.zeros_like(self.ei, dtype=np.int)
            for i,ei in enumerate(self.ei):lns[i] = len(self.dataset.x.items[self.idx[ei]])
            print( pd.DataFrame(data=np.stack([self.ei,self.eo,lns],axis=0).T,columns=["ei","eo","length"]) )
            self.sql = np.zeros((1,self.ite_len), dtype=np.int)
        else:
            self.ei,self.eo = 0,0

        i = 0
        countToks = 0
        while i < self.ite_len: 
            #load max batch first in order to reduce fragmentation i pytorch/GPU
            if self.first and i == 0: self.first,seq_len = False, int(self.buffer.size/self.bs - 1)
            else:                     seq_len = int( math.ceil(self.bptt*(1. + self.p_bptt*(np.random.random() - 0.5))) )

            nToks      = self.bs*(seq_len+1)
            countToks +=nToks #for printout

            if self.bl == BatchLayout.Parallel:
                data = self.buffer[:nToks].reshape(self.bs,-1)
                self.parallel_fill_buffer(data, self.ei,self.eo, overlap=1)
            else:    
                data = self.buffer[:nToks]
                self.ei, self.eo = self.fill_row(data,self.ei, self.eo, overlap=1)
                data = data.reshape(self.bs,-1)

            data  = torch.as_tensor( data, dtype=torch.long )
            res   = data[:,0:seq_len], data[:,1:seq_len+1]        
            i    += 1
            yield res  

        print(f"len(self):{len(self)} Number of iteration:{i}")
        print(f"\n\nself.ite_len:{self.ite_len} Number of iterations:{i} countToks:{countToks} self.totalToks:{self.totalToks} countToks < self.totalToks:{countToks < self.totalToks}")    
        for i,ei in enumerate(self.ei):lns[i] = len(self.dataset.x.items[self.idx[ei]])
        print( pd.DataFrame(data=np.stack([self.ei,self.eo,lns],axis=0).T,columns=["ei","eo","length"]) )
        print(f"\n\n")


    def fill_row(self, row, ei, eo, overlap):
        "new the tokens in the buffer with nToks from the ragged array"
        #nToks: number of tokens to be extract to row
        #ei:    index of the first rag to be extract. Returned as index to the next rag to be extracted
        #eo:    index to the first token to be extracted in the first rag. Returned pointing to the next 
        #       token to be extract in the last rag        
        ibuf, nToks = 0, row.size
        items = self.dataset.x.items
        bi,bo = ei,eo
        while nToks > ibuf:   
            rag   = items[self.idx[ei]]
            if ibuf==0: print( f"BEGIN: ei:{ei} eo:{eo} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} bi:{bi} bo:{bo} first toke:{rag[eo]}" )
            eo    = eo if ibuf==0 else 0         
            n     = min(len(rag) - eo, nToks - ibuf)
            print( f"ITE:  ei:{ei} eo:{eo} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo+n-1]}" )
            row[ibuf:ibuf+n] = rag[eo:eo+n]
            ibuf += n
            if ibuf < nToks: ei += 1
            elif overlap==1: 
                eo += n-overlap
                print(f"ENDB:ei:{ei} eo:{eo} nToks:{nToks} ibuf:{ibuf} bi:{bi} bo:{bo} next first tok:{rag[eo]}" )
            else: raise ValueError("overlap != 1 has not been implmented")

        return ei,eo

    def fill_row_2(self, row, ei, eo, overlap):
        "new the tokens in the buffer with nToks from the ragged array"
        #nToks: number of tokens to be extract to row
        #ei:    index of the first rag to be extract. Returned as index to the next rag to be extracted
        #eo:    index to the first token to be extracted in the first rag. Returned pointing to the next 
        #       token to be extract in the last rag        
        ibuf, j, nToks = 0, ei, row.size
        items = self.dataset.x.items
        bi,bo = ei,
        while nToks > ibuf:   
            rag = items[self.idx[j]]
            r0  = eo if ibuf==0 else 0         
            r1  = len(rag)
            n   = min(r1 - r0,nToks - ibuf)
            row[ibuf:ibuf+n] = rag[r0:r0+n]
            ibuf += n
            j    += 1
            if ibuf==nToks:
                #the following overlap is not general yet
                n -= overlap #the is an overlap of 1 token between the previous and the next sequence 

                #print(f"r0+n:{r0+n} len(rag):{len(rag)}")
                eo = 0 if r0+n==len(rag) else r0 + n 
                ei = j if r0+n==len(rag) else j  - 1 

            #print( f"j:{j} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} rl:{n} bi:{bi} bo:{bo} ei:{ei} eo:{eo}" )
        return ei,eo

    def parallel_fill_buffer(self, data, ei, eo, overlap):
        "data, ei, eo are passed by ref so updating then here updates the arrays in caller"
        for j in range(len(data)):
            #print(f"row:{j}")
            ei[j],eo[j] = self.fill_row( data[j], ei[j], eo[j], overlap)

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

    @property
    def batch_size(self): return self.bs
    @batch_size.setter
    def batch_size(self, v): self.bs = v

    def batchify(self, data:np.ndarray) -> LongTensor: pass

class MyTextLMDataBunch(TextLMDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None, 
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None, 
                 processor:PreProcessor=None, **kwargs) -> DataBunch:
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
