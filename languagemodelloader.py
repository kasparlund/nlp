from fastai.text import * 
def usedGB_RAM(): 
    import psutil
    return round((psutil.virtual_memory().used + psutil.swap_memory().used)/1e9,2)
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

    @staticmethod
    def testSaveIndexes(filepath,bs,eit,eot,idx,items):
        "save the index so that we can inspect them afterwards"
        print(eit)
        lengths = np.zeros_like(eit, dtype=np.int)
        for i in range(eit.shape[1]):
            for k in range(bs): 
                iidx = idx[eit[k,i]]
                lengths[k,i] = len(items[iidx])

        columns =       [f"length_bs{i}" for i in range(bs)]
        columns.extend( [f"ei_bs{i}"     for i in range(bs)])
        columns.extend( [f"eo_bs{i}"     for i in range(bs)])
        columns = np.asarray(columns)             
        data=np.concatenate([lengths,eit,eot],axis=0).T
        pf = pd.DataFrame(data=data,columns=columns)
        pf.to_csv(filepath,index=False)
        print(pf)
                
    def __init__(self, dataset:LabelList, bs:int=64, bptt:int=70, backwards:bool=False, shuffle:bool=False,
                 max_len:int=25, p_bptt:int=0.95, bl:BatchLayout=BatchLayout.Parallel):
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
        print(f"1 bl:{self.bl} __init__ self.bl is BatchLayout.Parallel.{self.bl == BatchLayout.Parallel}")

        #self.minToks = 4 #argument used to discard end of sections that are too short to be used for prediction of the next word
        print(f"LanguageModelLoader.__init__ Used GB memory:{usedGB_RAM()} batches:{len(self)} nToks:{nToks} "+\
              f"bptt:{self.bptt} p_bptt:{self.p_bptt} shuffle:{self.shuffle} backwards:{self.backwards}" )  


    def allocate_buffers(self):     
        "allocate the required worth-case batchbuffer"

        #CircularIndex is used to index the raggged dataset. It 1) can shuffled in-place, 
        # 2)can be configured to move backward, 3) it will warp around when the index 
        # exceeds the length of the underlying ragged array
        self.idx = MyLanguageModelLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)        

        #The batches vary uniformly around bppt as defined by p_bptt
        max_batch_element = self.bs * (1 + math.ceil( self.bptt*(1+0.5*self.p_bptt) ) )
        self.buffer       = np.zeros(max_batch_element, dtype=np.long)
        
        if self.bl == BatchLayout.Parallel:
            self.ei = np.zeros(self.bs, dtype=np.int)
            self.eo = np.zeros(self.bs, dtype=np.int)
        print(f"LanguageModelLoader.allocate_buffers Used GB memory:{usedGB_RAM()} "+\
              f"shuffle:{self.shuffle} backwards:{self.backwards}" )  
        
    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None: 
            yield LongTensor(getattr(self.dataset, 'item'))[None],LongTensor([0])

        #allocate buffers lazily in order to avoid vasting memory on fix_ds which is not always used/may never of ULMFit
        if self.idx is None: self.allocate_buffers()
        if self.shuffle: self.idx.shuffle()

        print(f"1 bl:{self.bl} __iter__ self.bl is BatchLayout.Parallel.{self.bl == BatchLayout.Parallel}")
        if self.bl == BatchLayout.Parallel:
            print(f"2 bl:{self.bl} __iter__ self.bl is BatchLayout.Parallel.{self.bl == BatchLayout.Parallel}")
            self.eo *= 0 
            step     = len(self.idx)//self.bs #step is truncated => batches may overlap a bit
            for i in range(self.bs): self.ei[i] = i*step
            print(f"self.ite_len:{self.ite_len} dataset.x.items:{len(self.dataset.x.items)}self.idx:{len(self.idx)} step:{step}")    
            print(f"self.ei:\n{self.ei}")
            #track the iterations for print
            self.eit = np.zeros((self.bs,self.ite_len), dtype=np.int)
            self.eot = np.zeros((self.bs,self.ite_len), dtype=np.int)
        else:
            self.ei,self.eo = 0,0

        i = 0
        while i < self.ite_len:
            self.i=i  
            seq_len = int(self.bptt*(1. + self.p_bptt*(np.random.random() - 0.5)))
            nToks   = self.bs*(seq_len+1)

            if self.bl == BatchLayout.Parallel:
                data = self.buffer[:nToks].reshape(self.bs,-1)
                self.parallel_fill_buffer(data, self.ei,self.eo)
                #for j in range(len(data)): self.ei[j],self.eo[j] = self.fill_row(data[j],self.ei[j],self.eo[j])
                self.eit[:,i] = self.ei[:]
                self.eot[:,i] = self.eo[:]

            else:    
                data = self.buffer[:nToks]
                self.ei, self.eo  = self.fill_row(data,self.ei, self.eo)
                data = data.reshape(self.bs,-1)

            data  = torch.as_tensor( data, dtype=torch.long )
            res   = data[:,0:seq_len-1], data[:,1:seq_len]        
            i    += 1
            #if i==self.ite_len : print(res) # check that y is shift to predict x       
            yield res  

            MyLanguageModelLoader.testSaveIndexes(Path.cwd()/"test.csv",self.bs,self.eit,self.eot,self.idx,self.dataset.x.items)


    def fill_row(self, row, ei,eo):
        "new the tokens in the buffer with nToks from the ragged array"
        #nToks: number of tokens to be extract and inserted starting at the beginning of the buffer from the 
        #       last saved indices in the ragged array and forward - possibly wrapping to the head of the dataset
        #ei: index of the last rag to be extract
        #eo: index (not inclusive) where the extract stops in the last rag
        #ei and eo starts at 0,0 in the beginning of a batch and is then updated as we fill in the buffer 
        #in the batch and batch by batch
        
        #ei,eo = ragidx[0],ragidx[1]
        ibuf, j, nToks = 0, ei, row.size
        items = self.dataset.x.items
        while nToks > ibuf:   
            i   = self.idx[j]
            rag = items[i]
            r0  = eo if j==ei else 0         
            r1  = len(rag)
            rl  = r1 - r0
            if ibuf+rl >= nToks:
                eo = (nToks + eo) if j==ei else (nToks-ibuf) 
                ei = j
                r1 = eo
                rl = r1 - r0
            row[ibuf:ibuf+rl] = rag[r0:r1]
            ibuf += rl
            j    += 1 
        #if self.ei==bi: print( f"one rag:{self.ei==bi} nToks:{nToks} nBToks:{ibuf} bi:{bi} bo:{bo} ei:{self.ei} eo:{self.eo}" 
        return ei,eo    

    def parallel_fill_buffer(self, data, ei, eo):
        for j in range(len(data)):
            ei[j],eo[j] = self.fill_row(data[j],ei[j],eo[j])

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
