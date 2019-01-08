from fastai.text import * 

class BatchLayout(IntEnum):
    Parallel   = 1
    Sequential = 2
    
class MyLanguageModelLoader():
    "Create a dataloader with bptt slightly changing."
    
    class CircularIndex():
        #When the index exceeds the length of self.idx then it is wrap to start at the head 
        #or the end if indexing is moving backwards. The shuffled is done in-place, 
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
                 p_bptt:int=0.0, bl:BatchLayout=BatchLayout.Parallel):
        self.init_kwargs = dict(bs=bs, bptt=bptt, backwards=backwards, shuffle=shuffle)
        self.dataset,self.bs,self.bptt,self.p_bptt,self.shuffle,self.backwards = dataset,bs,bptt,p_bptt,shuffle,backwards
        
        nToks = 0
        for i,s in enumerate(dataset.x.items): nToks += len(s)
        self.totalToks   = nToks    
        self.ite_len     = math.ceil( nToks / (self.bs*self.bptt) )
        self.idx         = None
        self.npStorage   = None
        self.bl          = bl        
        self.first       = True
        self.num_workers = 0 # how is num_workers used here ?

    def __len__(self) -> int: return self.ite_len

    def allocate_buffers(self):     
        "allocate the required worth-case batchbuffer"
        self.idx = MyLanguageModelLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)        

        #The batches vary uniformly around bppt in the interval bppt + p_bptt*bppt
        max_batch_element = int( self.bs * (1 + math.ceil( self.bptt*(1+0.5*self.p_bptt) ) ) )
        self.npStorage    = np.zeros(max_batch_element, dtype=np.int64)
        
        if self.bl == BatchLayout.Parallel:
            self.ei = np.zeros(self.bs, dtype=np.int)
            self.eo = np.zeros(self.bs, dtype=np.int)

    def print_ei_eo(self, title:str):
        #usefull for verification that each row uses all its data
        lns = np.zeros_like(self.ei, dtype=np.int)
        for i,ei in enumerate(self.ei):lns[i] = len(self.dataset.x.items[self.idx[ei]])
        print(title)
        print( pd.DataFrame(data=np.stack([self.ei,self.eo,lns],axis=0).T,columns=["ei","eo","length"]) )

    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None: 
            yield LongTensor(getattr(self.dataset, 'item'))[None],LongTensor([0])

        #allocate buffers lazily in order to avoid vasting memory on fix_ds which is not always or never for ULMFit
        if self.idx is None: self.allocate_buffers()
        if self.shuffle:     self.idx.shuffle()
        if self.backwards: raise ValueError("backwards is not been implmented yet")

        if self.bl == BatchLayout.Parallel:
            #Runs throurh the rags and set an offset where each row in the batch begins
            stepTokens = self.totalToks/self.bs 
            self.ei[0] = self.eo[0] = i_row = countTokens = 0
            for i in range( len( self.dataset.x.items) ):
                countTokens += len( self.dataset.x.items[self.idx[i]] )
                while countTokens > int( (i_row+1) * stepTokens) and i_row+1 < self.bs:
                    i_row         += 1
                    self.ei[i_row] = i
                    self.eo[i_row] = countTokens - int(i_row*stepTokens) - 1
            #self.print_ei_eo("start of epoch")
        else:
            self.ei,self.eo = 0,0

        i = 0
        countToks = 0
        while i < self.ite_len: 
            #load max batch first, in order to reduce fragmentation i pytorch/GPU!
            if self.first and i == 0: self.first,seq_len = False, int(self.npStorage.size/self.bs - 1)
            else:                     seq_len = int( math.ceil(self.bptt*(1. + self.p_bptt*(np.random.random() - 0.5))) )
            nToks = self.bs*(seq_len+1)
            countToks+=nToks   

            if self.bl == BatchLayout.Parallel:
                batchView = self.npStorage[:nToks].reshape(self.bs,-1)
                self.parallel_fill_buffer(self.dataset.x.items, self.idx, batchView, self.ei, self.eo, overlap=1)
            else:    
                batchView = self.npStorage[:nToks]
                self.ei, self.eo = self.fill_row(self.dataset.x.items, self.idx, batchView, self.ei, self.eo, overlap=1)
                batchView = batchView.reshape(self.bs,-1)

            i += 1
            yield torch.from_numpy(batchView[:,0:seq_len]), torch.from_numpy(batchView[:,1:seq_len+1])
        #self.print_ei_eo("end of epoch")
        print(f"totalToks:{self.totalToks} processedTokens:{countToks} processedTokens-totalToks:{countToks-self.totalToks}")

    def fill_row(self, items, idx, row, ei, eo, overlap):
        "new the tokens in the buffer with nToks from the ragged array"
        #items:   the ragged/jagged array 
        #idx:     used to index into items so that shuffle is used - if idx has been shuffled 
        #row:     a row in teh batch to be filled with consequetive tokens
        #ei:      index of the first rag to be extract. Returs as index to the next rag to be extracted
        #eo:      index to the first token to be extracted in the first rag. Returs pointing to the next to be extract in the last rag
        #overlap: overlap=1 between batches, because we only predict the next token

        #bi,bo = ei,eo
        #print(f"BEGIN:rowid:{rowid} ei:{ei} eo:{eo} nToks:{nToks} bi:{bi} bo:{bo}" )
        ibuf = 0
        while ibuf < row.size:   
            rag   = items[idx[ei]]
            #if ibuf==0: print( f"BEGIN: ei:{ei} eo:{eo} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} bi:{bi} bo:{bo} first toke:{rag[eo]}" )
            eo    = eo if ibuf==0 else 0         
            n     = min(len(rag) - eo, row.size - ibuf)
            #print( f"ITE:  ei:{ei} eo:{eo} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo+n-1]}" )
            row[ibuf:ibuf+n] = rag[eo:eo+n]
            ibuf += n
            if ibuf < row.size: ei += 1
            elif overlap == 1:  eo += n-overlap
                #print(f"ENDB:ei:ei:{ei} eo:{eo} len(rag):{len(rag)} nToks:{nToks} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo]}" )
            else: raise ValueError("overlap != 1 has not been implmented")

        return ei,eo

    def parallel_fill_buffer(self, items, idx, batch, ei, eo, overlap):
        "data, ei, eo are passed by ref so updating then here updates the arrays in caller"
        for j in range(len(batch)):
            ei[j],eo[j] = self.fill_row( items, idx, batch[j], ei[j], eo[j], overlap )

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

    @property
    def batch_size(self): return self.bs
    @batch_size.setter
    def batch_size(self, v): self.bs = v

    def batchify(self, data:np.ndarray) -> LongTensor: pass


#this class is only to ensure that MyLanguageModelLoader gets loaded instead of LanguageModelLoader
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
