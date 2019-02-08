from fastai.text import * 

class BatchLayout(IntEnum):
    Parallel   = 1
    Sequential = 2

class MyLanguageModelPreLoader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length),forward
        def __getitem__(self, i): return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self,ro=None):
            "shuffle CircularIndex indicies and new indices that points to the jagged arrays that ro pointed to indirectly"
            if ro is None: np.random.shuffle(self.idx)
            else:          
                t0 = time.perf_counter()
                log = False
                if log:idx_cpy = self.idx.copy()

                #get the location(ro_idx) of the data indices for ro and the data indices themself (rod)
                ro_ix  = ro%len(self.idx) if self.forward else len(self.idx)-1-ro%len(self.idx)
                rod    = self.idx[ro_ix]
            
                #remove rod from idx before shuffle
                idx_rp = np.delete(self.idx,ro_ix)
                if len(rod)+len(idx_rp) == len(self.idx):                     
                    #we can proceed with NO ties(ie no ro_idx points to the same value in idx)

                    #shuffle the remaining indicies in idx
                    np.random.shuffle(idx_rp) 

                    #insert rod with equal intervals in the shuffled idx
                    step     = len(idx_rp) / len(rod)
                    offsets  = np.round(np.arange(len(rod))*step).astype(np.int)
                    self.idx = np.insert(idx_rp, offsets, rod)
                    ro_new   = offsets+np.arange(len(offsets))

                    if log:
                        ro_after = self.idx[ ro_new%len(self.idx) if self.forward else len(self.idx)-1-ro_new%len(self.idx)]
                        if not ((ro_after-rod)==0).all():
                            print(f"\nstep  :{step}\nidx_rp:{idx_rp}\noffs  :{offsets}")
                            print(f"\n\nidx      :{self.idx}")
                            print( f"\ndiff     :{(ro_after-rod)}\nro_after :{ro_after}\nro_before:{rod}" )

                        #print(f"idx      :{self.idx}")
                        #print(f"idx_cpy  :{idx_cpy}")
                        #pct_changed = (len(self.idx) - np.sum((self.idx-idx_cpy)==0))/len(self.idx)
                        #print(f"Time to shuffle: {time.perf_counter()-t0} sec. Shuffled {len(idx_rp)}/{len(self.idx)} tokens => changed:{int(pct_changed*100.+.5)} %. ")        
                    #print(f"Time to shuffle: {(time.perf_counter()-t0):.1e}")        
                else:
                    #print("no shuffle")
                    #we do not shuffle when there is ties because ths only occure in tiny datasets such as testdata
                    ro_new = ro
            return ro_new

    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False, 
                 shuffle:bool=False):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.totalToks,self.ite_len,self.idx = int(0),None,None

    def __len__(self): 
        if self.ite_len is None:
            if self.lengths is None: self.lengths = np.fromiter( (len(item) for item in self.dataset.x.items),dtype=np.int,count=len(self.dataset.x.items) )
            self.totalToks = self.lengths.sum()
            self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        return self.ite_len

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)
   
    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None: len(self)
        self.idx   = LanguageModelPreLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)
        self.batch = np.zeros((self.bs, self.bptt+1), dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,1:self.bptt+1] 
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        self.ri    = np.zeros(self.bs, dtype=np.int)

        t0 = time.perf_counter()
        step = self.totalToks / self.bs
        ln_rag, countTokens, i_rag = 0, 0, -1
        for i in range(0,self.bs):
            #Compute the initial values for ro and ri 
            while ln_rag + countTokens <= int(step * i):
                countTokens += ln_rag
                i_rag       += 1
                ln_rag       = self.lengths[self.idx[i_rag]]
            self.ro[i] = i_rag
            self.ri[i] = ( ln_rag - int(step * i - countTokens) ) if self.backwards else int(step * i - countTokens)
        #print(f"time to initialize ro,ri:{(time.perf_counter()-t0):.1e}")

    def printJagged(self):
        for j in range(len(self.dataset.x.items)): print(f"r{j},{self.idx[j]} :{self.dataset.x.items[self.idx[j]]}")
    def on_epoch_begin(self, **kwargs):
        #after the first epoch get the direct location of ro in the source data 
        #ro_from = None if self.idx is None else  [self.idx[i] for i in self.ro]
        if self.idx is None: self.allocate_buffers()
        elif self.shuffle: 
            ro_before = np.fromiter((self.idx[r] for r in self.ro), dtype=np.int, count=len(self.ro))
            self.ro = self.idx.shuffle(self.ro)
            ro_after  = np.fromiter((self.idx[r] for r in self.ro), dtype=np.int, count=len(self.ro))
            assert ((ro_after-ro_before)==0).all(), f"\nfailed   :{(ro_after-ro_before)}\nro_after :{ro_after}\nro_before:{ro_before}"

        self.idx.forward = not self.backwards 
         
    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): self.on_epoch_begin()

    def __getitem__(self, k:int):
        j = k % self.bs
        if j==0:
            if self.item is not None: return self.dataset[0]
            if self.idx is None: self.on_epoch_begin()
        self.ro[j],self.ri[j] = self.fill_row(not self.backwards, self.dataset.x.items, self.idx, self.batch[j], 
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j], self.batch_y[j]

    def fill_row(self, forward, items, idx, row, ro, ri, overlap,lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        #print(f"B: ro:{ro} ri:{ri}´")
        ibuf = n = 0 
        ro  -= 1
        while ibuf < row.size:  
            ro   += 1 
            ix    = idx[ro]
            rag   = items[ix]
            if forward:
                ri = 0 if ibuf else ri
                n  = min(lengths[ix] - ri, row.size - ibuf)
                row[ibuf:ibuf+n] = rag[ri:ri+n]
            else:    
                ri = lengths[ix] if ibuf else ri
                n  = min(ri, row.size - ibuf) 
                row[ibuf:ibuf+n] = rag[ri-n:ri][::-1]
            ibuf += n
        ro,ri = ro, ri + ((n-overlap) if forward else -(n-overlap))
        #print(f"E: ro:{ro} ri:{ri}´")
        return ro,ri

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