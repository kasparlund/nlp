from fastai.text import * 

class BatchLayout(IntEnum):
    Parallel   = 1
    Sequential = 2
    
class MyLanguageModelPreLoader(Callback):
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

    def __init__(self, dataset:LabelList, bs:int=32, bptt:int=70, backwards:bool=False, shuffle:bool=False,
                 drop_last:bool=False, bl:BatchLayout=BatchLayout.Parallel, log=False):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards = dataset,bs,bptt,shuffle,backwards
        self.totalToks = 0
        for rag in dataset.x.items: self.totalToks += len(rag)
        self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        self.npStorage,self.idx, self.bl, self.log = None,None, bl, log
        if self.log:print(f"MyLanguageModelPreLoader.__init__ totalToks:{self.totalToks} ite_len:{self.ite_len} bs:{self.bs} bptt:{self.bptt}")

    def __len__(self): return self.ite_len
    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)
   
    def allocate_buffers(self):     
        "allocate the required worth-case batchbuffer"
        #items:   the ragged/jagged array 
        #idx:     used to index into items so that shuffle is used - if idx has been shuffled 
        #row:     a row in the batch to be filled with consequetive tokens
        #ei:      index of the first rag to be extract. Returs as index to the next rag to be extracted
        #eo:      index to the first token to be extracted in the first rag. Returs pointing to the next to be extract in the last rag
        #         when iterating backwards then ei becomes 1+ the last token to be extraced in the rag 
        #overlap: overlap=1 between batches, because we only predict the next token        
        self.idx   = MyLanguageModelPreLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)
        self.batch = np.zeros( (self.bs, self.bptt+1), dtype=np.int64)
        self.x, self.y = self.batch[:,0:self.bptt], self.batch[:,1:self.bptt+1]      
        self.ei    = np.zeros(self.bs, dtype=np.int) if self.bl == BatchLayout.Parallel else 0
        self.eo    = np.zeros(self.bs, dtype=np.int) if self.bl == BatchLayout.Parallel else 0        

    def print_ei_eo(self, title:str):
        lns = np.zeros_like(self.ei, dtype=np.int)
        for i,ei in enumerate(self.ei):lns[i] = len(self.dataset.x.items[self.idx[ei]])
        print(title)
        print( pd.DataFrame(data=np.stack([self.ei,self.eo,lns],axis=0).T,columns=["ei","eo","length"]) )
    
    def on_epoch_begin(self, **kwargs):
        #print(f"MyLanguageModelPreLoader.on_epoch_begin bs:{self.bs} len(self):{len(self)} shuffle:{self.shuffle}")
        if self.idx is None: self.allocate_buffers()
        if self.shuffle:     self.idx.shuffle()
        self.idx.forward(self.backwards is False) 

        if self.bl == BatchLayout.Parallel:
            #set up ei and eo to index the data so batches has continuous rows  
            step   = self.totalToks / self.bs
            ln_rag = countTokens = 0
            i_rag  = -1
            if self.log: print(f"step:{step}")
            for i in range(0,self.bs):
                while ln_rag <= int(step * i) - countTokens :
                    countTokens += ln_rag
                    i_rag       += 1
                    ln_rag       = len( self.dataset.x.items[self.idx[i_rag]] )
                self.ei[i] = i_rag
                self.eo[i] = ( ln_rag - int(step * i - countTokens) ) if self.backwards else int(step * i - countTokens)

                if self.log: 
                    print(f"i_rag:{i_rag} ln_rag:{ln_rag} int(step * i):{int(step * i)} countTokens:{countTokens} self.eo[i]:{self.eo[i]}")    
                    self.print_ei_eo("start of epoch")
        else:
            self.ei,self.eo = 0,0
        self.countToks=0
    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): 
        self.on_epoch_begin()

    def __getitem__(self, k:int):
        if self.item is not None: return self.dataset[0]
        if self.idx is None:      self.on_epoch_begin()

        j = k % self.bs

        if self.bl == BatchLayout.Parallel:
            if self.backwards: 
                self.ei[j],self.eo[j] = self.fill_backward( self.dataset.x.items, self.idx, self.batch[j], 
                                                            self.ei[j], self.eo[j], overlap=1, rowid=j )
            else:         
                self.ei[j],self.eo[j] = self.fill_forward(  self.dataset.x.items, self.idx, self.batch[j], 
                                                            self.ei[j], self.eo[j], overlap=1, rowid=j )
        else:    
            if self.backwards: 
                self.ei, self.eo = self.fill_backward(      self.dataset.x.items, self.idx, self.batch.flatten(), 
                                                            self.ei, self.eo, overlap=1, rowid=j )
            else:              
                self.ei, self.eo = self.row_fill(           self.dataset.x.items, self.idx, self.batch.flatten(),
                                                            self.ei, self.eo, overlap=1, rowid=j )
        self.countToks += self.bptt
        #return self.batch[j,0:self.bptt], self.batch[j,1:self.bptt+1]
        return self.x[j], self.y[j]

    def fill_forward(self, items, idx, row, ei, eo, overlap,rowid):
        "fill the row with tokens reading forwards from the ragged array"
        ibuf = 0
        ei  -= 1 
        while ibuf < row.size:  
            ei   += 1 
            rag   = items[idx[ei]]
            eo    = eo if ibuf==0 else 0
            n     = min(len(rag) - eo, row.size - ibuf)
            row[ibuf:ibuf+n] = rag[eo:eo+n]
            ibuf += n
        if overlap == 1:  eo += n-overlap
        else: raise ValueError("overlap != 1 has not been implemented")

        return ei,eo

    def fill_backward(self, items, idx, row, ei, eo, overlap,rowid):
        "fill the row with tokens reading backwards from the ragged array"
        ibuf = 0
        ei  -= 1 
        while ibuf < row.size:  
            ei   += 1 
            rag   = items[idx[ei]]
            eo    = eo if ibuf==0 else len(rag)
            n     = min(eo, row.size - ibuf) 
            row[ibuf:ibuf+n] = rag[eo-n:eo][::-1]
            ibuf += n
        if overlap == 1: eo -= n-overlap
        else: raise ValueError("overlap != 1 has not been implemented")

        return ei,eo

#this class is only to ensure that MyLanguageModelLoader gets loaded instead of LanguageModelLoader
class MyTextLMDataBunch(TextLMDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None,
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None,
                 processor:PreProcessor=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a `vocab`."
        src = ItemLists(path, TextList(train_ids, vocab, path=path, processor=[]),
                        TextList(valid_ids, vocab, path=path, processor=[]))
        src = src.label_for_lm() if cls==MyTextLMDataBunch else src.label_from_lists(train_lbls, valid_lbls, classes=classes, processor=[])
        if test_ids is not None: src.add_test(TextList(test_ids, vocab, path=path), label=train_lbls[0])
        src.valid.x.processor = ifnone(processor, [TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        src.train.x._bunch = MyTextLMDataBunch
        src.valid.x._bunch = MyTextLMDataBunch
        if src.test is not None: src.test.x._bunch  = MyTextLMDataBunch
        return src.databunch(**kwargs)

    #need customized version of this in order to set MyLanguageModelLoader
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, num_workers:int=0,
               device:torch.device=None, collate_fn:Callable=data_collate, tfms:Optional[Collection[Callable]]=None, 
               **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        datasets = [MyLanguageModelPreLoader(ds, shuffle=(i==0), drop_last=(i==0), bs=bs, **kwargs) for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn, no_check=no_check)