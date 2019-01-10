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
                 p_bptt:int=0.0, bl:BatchLayout=BatchLayout.Parallel, log=False):
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
        self.log         = log

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
    import pdb
    def __iter__(self):
        if getattr(self.dataset, 'item', None) is not None: 
            yield LongTensor(getattr(self.dataset, 'item'))[None],LongTensor([0])

        #allocate buffers lazily in order to avoid vasting memory on fix_ds which is not always or never for ULMFit
        if self.idx is None: self.allocate_buffers()
        if self.shuffle:     self.idx.shuffle()
        self.idx.forward(self.backwards is False) 

        if self.bl == BatchLayout.Parallel:
            stepTokens = self.totalToks / self.bs
            ln_rag = countTokens = 0
            i_rag  = -1
            if self.log: print(f"stepTokens:{stepTokens}")
            for i in range(0,self.bs):
                while ln_rag <= int(stepTokens * i) - countTokens :
                    countTokens += ln_rag
                    i_rag       += 1
                    ln_rag       = len( self.dataset.x.items[self.idx[i_rag]] )

                self.ei[i] = i_rag
                self.eo[i] = ln_rag - int(stepTokens * i - countTokens) if self.backwards else int(stepTokens * i - countTokens) 
                if self.log: print(f"i_rag:{i_rag} ln_rag:{ln_rag} int(stepTokens * i):{int(stepTokens * i)} countTokens:{countTokens} self.eo[i]:{self.eo[i]} ")    
            #self.print_ei_eo("start of epoch")
        else:
            self.ei,self.eo = 0,0

        i = 0
        overlap=1
        while i < self.ite_len: 
            #load max batch first, in order to reduce fragmentation i pytorch/GPU!
            if self.first and i == 0: self.first,seq_len = False, int(self.npStorage.size/self.bs - 1)
            else:                     seq_len = int( math.ceil(self.bptt*(1. + self.p_bptt*(np.random.random() - 0.5))) )
            nToks = self.bs*(seq_len+1)

            if self.bl == BatchLayout.Parallel:
                batchView = self.npStorage[:nToks].reshape(self.bs,-1)
                self.parallel_fill(self.dataset.x.items, self.idx, batchView, self.ei, self.eo, overlap, self.backwards)
            else:    
                batchView = self.npStorage[:nToks]
                if  backwards:self.ei, self.eo = self.fill_row_backwards(self.dataset.x.items,self.idx, batchView,\
                                                               self.ei, self.eo, overlap)
                else:         self.ei, self.eo = self.fill_row(self.dataset.x.items, self.idx, batchView,\
                                                               self.ei, self.eo, overlap)
                batchView = batchView.reshape(self.bs,-1)

            i += 1
            yield torch.from_numpy(batchView[:,0:seq_len]), torch.from_numpy(batchView[:,1:seq_len+1])
        #self.print_ei_eo("end of epoch")

    def fill_forward(self, items, idx, row, ei, eo, overlap,rowid):
        "fill the row with tokens reading forwards from the ragged array"
        #items:   the ragged/jagged array 
        #idx:     used to index into items so that shuffle is used - if idx has been shuffled 
        #row:     a row in teh batch to be filled with consequetive tokens
        #ei:      index of the first rag to be extract. Returs as index to the next rag to be extracted
        #eo:      index to the first token to be extracted in the first rag. Returs pointing to the next to be extract in the last rag
        #overlap: overlap=1 between batches, because we only predict the next token
        bi,bo = ei,eo
        #print(f"BEGIN:rowid:{rowid} ei:{ei} eo:{eo} row.size:{row.size} bi:{bi} bo:{bo}" )
        ibuf = 0
        ei  -= 1 
        while ibuf < row.size:  
            ei   += 1 
            rag   = items[idx[ei]]
            #if ibuf==0: print( f"BEGIN: ei:{ei} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} bi:{bi} bo:{bo} first toke:{rag[eo]}" )
            eo    = eo if ibuf==0 else 0
            n     = min(len(rag) - eo, row.size - ibuf)
            #print( f"ITE:  ei:{ei} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo+n-1]}" )
            row[ibuf:ibuf+n] = rag[eo:eo+n]
            ibuf += n
        if overlap == 1:  
            eo += n-overlap
            #print(f"ENDB:ei:ei:{ei} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo]}" )
        else: raise ValueError("overlap != 1 has not been implemented")

        return ei,eo

    def fill_backward(self, items, idx, row, ei, eo, overlap,rowid):
        "fill the row with tokens reading backwards from the ragged array"
        bi,bo = ei,eo
        ibuf = 0
        ei  -= 1 
        while ibuf < row.size:  
            ei   += 1 
            i     = idx[ei]
            rag   = items[idx[ei]]
            if ibuf==0: print( f"BEGIN:ei:{ei} i:{i} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} bi:{bi} bo:{bo} first toke:{rag[eo-1]}" )
            eo    = eo if ibuf==0 else len(rag)
            n     = min(eo, row.size - ibuf) 
            print( f"ITE:  ei:{ei} i:{i} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} last toke:{rag[eo-n-1]}" )
            row[ibuf:ibuf+n] = rag[eo-n:eo][::-1]
            ibuf += n
        if overlap == 1:  
            if n == 1: ei -= 1
            else     : 
                eo -= n-overlap
                print(f"ENDB:ei:ei:{ei} i:{i} eo:{eo} len(rag):{len(rag)} row.size:{row.size} ibuf:{ibuf} n:{n} bi:{bi} bo:{bo} ENDB2:last toke:{rag[eo-1]}")
        else: raise ValueError("overlap != 1 has not been implemented")

        return ei,eo

    def row_fill(self, items, idx, batch, ei, eo, overlap, backwards):
        if backwards:self.ei, self.eo = self.fill_forward( self.dataset.x.items, self.idx, batchView, self.ei, self.eo, overlap)
        else:        self.ei, self.eo = self.fill_backward(self.dataset.x.items, self.idx, batchView, self.ei, self.eo, overlap)

    def parallel_fill(self, items, idx, batch, ei, eo, overlap, backwards):
        "data, ei, eo are passed by ref so updating then here updates the arrays in caller"
        if backwards: 
            for j in range(len(batch)): ei[j],eo[j] = self.fill_backward( items, idx, batch[j], ei[j], eo[j], overlap, j )
        else:         
            for j in range(len(batch)): ei[j],eo[j] = self.fill_forward(  items, idx, batch[j], ei[j], eo[j], overlap, j )

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
