from fastai.text import * 

def random_bucket(nBuckets):
    nRepeat,ix_last = int(1e4)//nBuckets,0
    numbers = np.repeat(np.arange(nBuckets,dtype=np.int),nRepeat)
    def rand_int():
        nonlocal numbers,ix_last
        if ix_last == 0:
            ix_last = len(numbers)
            np.random.shuffle(numbers)
        ix_last -= 1    
        return numbers[ix_last]
    return rand_int

class ShuffleSampler(Sampler):
    "samples index from range 0-length infinitely by shuffle the index when they have all been used"
    def __init__(self, length:int, shuffle:bool=True): 
        self.idx, self.ix_last, self.shuffle = np.arange(length), length, shuffle
        super().__init__(self.idx)
    def __len__(self) -> int: return len(self.idx)
    def __iter__(self): return self    
    def __next__(self):        
        ix = self.ix_last = self.ix_last - 1
        if ix == -1: 
            ix = self.ix_last = len(self.idx) - 1
            if self.shuffle:np.random.shuffle(self.idx)
        return self.idx[ix]
    def sample(self, n:int ): return np.fromiter( (s for s in self), dtype=np.int64, count=n)

class SampleOnLength(Sampler):
        
    def __init__(self, lengths, bins, sampler:Sampler, min_bucket_size=1000):
        super().__init__(lengths)        
        self.lengths, self.bins, self.sampler = lengths, bins, sampler
        self.from_fixed_step(lengths,bins,min_bucket_size)
        
    def from_fixed_step(self, lengths, bins,min_bucket_size):
        #Group sentences by length in buckets with a fixed width(step). Then merge neighboring buckets 
        #so all contain at least min_bucket_size sentences  
        n_buckets, lower,upper,step = len(bins), bins[0],bins[-1],(bins[-1]-bins[0])/(len(bins)-1)
        
        buckets = np.empty(n_buckets,dtype=object)
        for i in range(n_buckets): buckets[i]=[]
        
        for i in range(len(lengths)): 
            length = lengths[i]-lower
            ib     = 0 if length<0 else (n_bucket-1 if length>upper else int(length/step) )
            buckets[ib].append(i)
            
        #merge buckets below min_bucket_size
        i=0
        while i < len(buckets):
            if len(buckets[i]) < min_bucket_size and len(buckets[i])>0:
                b = buckets[i]
                i += 1
                while len(b) < min_bucket_size and i < len(buckets):
                    if len(buckets[i]) > 0:
                        extend_by = min( len(buckets[i]), min_bucket_size-len(b) )
                        b.extend(buckets[i][:extend_by])
                        buckets[i] = buckets[i][extend_by:]
                    if len(buckets[i])==0 : i += 1
            else:       
                i += 1
        
        ix_empty = 0==np.fromiter( (len(b) for b in buckets), dtype=np.int, count=n_buckets)
        buckets  = buckets[ix_empty==False] 
        if len(buckets[-1]) < min_bucket_size:
            buckets[-2].extend(buckets[-1])
            buckets = buckets[:-1]
        
        #convert to arrays
        for i in range(len(buckets)): buckets[i] = np.asarray(buckets[i],np.int64)
        self.buckets = buckets
        self.ix_last = np.fromiter( (len(b) for b in buckets), dtype=np.int64, count=len(buckets) )
        
    def __len__(self) -> int: return len(self.lengths)
    def __iter__(self): return self    
    def __next__(self):        
        i_bucket = self.sampler()
        ix = self.ix_last[i_bucket] = self.ix_last[i_bucket] - 1        
        if ix == -1:
            ix = self.ix_last[i_bucket] = len(self.buckets[i_bucket])-1
            np.random.shuffle(self.buckets[i_bucket])
        return self.buckets[i_bucket][ix]
    def sample(self, n:int ): return np.fromiter( (s for s in self), dtype=np.int64, count=n)

class BatchLayout(IntEnum):
    Parallel   = 1
    Sequential = 2

class MyLanguageModelPreLoader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    
    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False, 
                 shuffle:bool=False):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.totalToks,self.ite_len,self.idx = int(0),None,None
        print("MyLanguageModelPreLoader")

    def __len__(self): 
        if self.ite_len is None:
            if self.lengths is None: self.lengths = np.fromiter( (len(item) for item in self.dataset.x.items),dtype=np.int,count=len(self.dataset.x.items) )
            self.totalToks = np.sum(self.lengths, dtype=np.int64 )
            self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        return self.ite_len

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)
   
    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None: len(self)

        self.resampleLengths = True
        if self.resampleLengths and self.shuffle:
            lower, upper, step = np.min(self.lengths), np.max(self.lengths), 10
            if (upper-lower)//step < 10: step = 1
            bins         = np.arange( lower, upper+step, step)
            #print(f"lower:{lower}, upper:{upper}, step:{step}\nbins:{bins}")
            uniform_sampler = random_bucket(nBuckets=len(bins))
            self.idx        = SampleOnLength(self.lengths, bins, uniform_sampler,min_bucket_size=2)
            self.idx.sampler = random_bucket(nBuckets=len(self.idx.buckets))
            #print(f"self.idx.buckets:{self.idx.buckets}")
        else:
            self.idx   = ShuffleSampler(len(self.dataset.x.items), self.shuffle)
        print(f"allocate_buffers sampler classs:{self.idx.__class__}")
        self.batch = np.zeros((self.bs, self.bptt+1), dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,1:self.bptt+1] 
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.fromiter( (next(self.idx) for i in range(self.bs)), dtype=np.int64, count=self.bs)
        #self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        #self.ri    = np.zeros(self.bs, dtype=np.int)
        self.ri    = np.fromiter( (self.lengths[self.ro[i]] if self.backwards else 0  for i in range(self.bs)), 
                                  dtype=np.int64, count=self.bs)

        """
        #t0 = time.perf_counter()
        #print(f"time to initialize ro,ri:{(time.perf_counter()-t0):.1e}")
        """

    def printJagged(self):
        for j in range(len(self.dataset.x.items)): print(f"r{j},{self.idx[j]} :{self.dataset.x.items[self.idx[j]]}")
    def on_epoch_begin(self, **kwargs):
        #after the first epoch get the direct location of ro in the source data 
        #ro_from = None if self.idx is None else  [self.idx[i] for i in self.ro]
        if self.idx is None: self.allocate_buffers()
        print(f"on_epoch_begin sampler classs:{self.idx.__class__}")
         
    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): self.on_epoch_begin()

    def __getitem__(self, k:int):
        j = k % self.bs
        if j==0:
            if self.item is not None: return self.dataset[0]
            if self.idx is None: self.on_epoch_begin()
        self.ro[j],self.ri[j] = self.fill_row(self.idx, not self.backwards, self.dataset.x.items, self.idx, self.batch[j], 
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j], self.batch_y[j]

    def fill_row(self, sampler, forward, items, idx, row, ro, ri, overlap,lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        #print(f"B: ro:{ro} ri:{ri}")
        ibuf = n = 0 
        #ro  -= 1
        while ibuf < row.size:  
            ro    = next(sampler) if ibuf else ro
            rag   = items[ro]
            if forward:
                ri = 0 if ibuf else ri
                n  = min(lengths[ro] - ri, row.size - ibuf)
                row[ibuf:ibuf+n] = rag[ri:ri+n]
            else:    
                ri = lengths[ro] if ibuf else ri
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