from fastai.text import * 

class Sampler():
    def __init__(self,populationSize:int):self.n_population = populationSize
    def sampleOne(self)->int: pass
    def changeSampler( self, sampler): self.sampler = sampler
    def __len__(self) -> int: return self.n_population
    
    class Iter():
        def __init__(self,sampler:Sampler,hasStop:bool, ix_first):
            self.sampler, self.i, self.hasStop = sampler, ix_first-1, hasStop
        def __next__(self):
            "Return next sample."
            #if self.i >= len(self.sampler) and self.hasStop: raise StopIteration
            #else: return self.sampler.sampleOne(self.i)
            #"""
            #i = self.i
            while True:
                yield self.sampler.sampleOne()
                #i += 1
                #yield self.sampler.sampleOne(i)
            #"""    
            #self.i += 1
            #return self.sampler.sampleOne()
            #return self.sampler.sampleOne(self.i)
            
    def __iter__(self,hasStop=True,ix_first:int=0): return Sampler.Iter(self,hasStop,ix_first)
    def sample(self, n:int ): 
        return np.fromiter( (self.sampleOne() for i in range(n)), dtype=np.int64, count=n)
        #i = iter(self)
        #return np.fromiter( (next(i)), dtype=np.int64, count=n)
        #return np.fromiter( (s for s in self), dtype=np.int64, count=n)
                
def random_bucket(nBuckets):
    nRepeat,ix_last = int(2.5e4)//nBuckets,0
    numbers = np.repeat(np.arange(nBuckets,dtype=np.int),nRepeat)
    #print(f"nRepeat:{nRepeat}\nlen(numbers):{len(numbers)}\nnumbers:{numbers}")
    def rand_int():
        nonlocal numbers,ix_last
        if ix_last == 0:
            ix_last = len(numbers)
            np.random.shuffle(numbers)
        ix_last -= 1    
        return numbers[ix_last]
    return rand_int

class BatchLayout(IntEnum):
    Parallel   = 1
    Sequential = 2

class MyLanguageModelPreLoader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    """        
    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length),forward
        def __getitem__(self, i): return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)
        def shuffle_old(self,ro=None):
            "shuffle CircularIndex indicies and new indices that points to the jagged arrays that ro pointed to indirectly"
            if ro is None: np.random.shuffle(self.idx)
            else:          
                #get the location(ro_idx) of the data indices for ro and the data indices themself (rod)
                ro_ix  = ro%len(self.idx) if self.forward else len(self.idx)-1-ro%len(self.idx)
                rod    = self.idx[ro_ix]
            
                #remove rod from idx before shuffle
                idx_rp = np.delete(self.idx,ro_ix)
                if len(rod)+len(idx_rp) == len(self.idx):                     
                    #we can proceed with NO ties(ie no ro_idx points to the same value in idx)
                    np.random.shuffle(idx_rp) 

                    #insert rod with equal intervals in the shuffled idx
                    step     = len(idx_rp) / len(rod)
                    offsets  = np.round(np.arange(len(rod))*step).astype(np.int)

                    self.idx = np.insert(idx_rp, offsets, rod)
                    ro_new   = offsets+np.arange(len(offsets))
                    if not self.forward:ro_new = len(self.idx)-1-ro_new%len(self.idx)
                else:
                    #we do not shuffle when there is ties because ths only occure in tiny datasets such as testdata
                    ro_new = ro
            return ro_new

        def shuffle(self,ro=None):
            "shuffle CircularIndex indicies and new indices that points to the jagged arrays that ro pointed to"
            if ro is None: np.random.shuffle(self.idx)
            elif len(np.unique(ro)) < len(ro) : return ro  #don't shuffle if we got ties (happens in tiny datasets)
            else: 
                #get indicies and values for ro
                ro_ix   = ro%len(self.idx) if self.forward else len(self.idx)-1-ro%len(self.idx)
                ro      = self.idx[ro_ix].flatten()

                #calc offset to insert ro in idx at equally spaced intervals
                step    = len(self.idx)/len(ro)
                offsets = np.round(np.arange(len(ro))*step).astype(np.int)

                #tag the positions for ro before shuffle and then find them again after shuffle
                self.idx[ro_ix] = -1
                np.random.shuffle(self.idx)
                ro_ix = self.idx == -1  #boolean with true in position with value -1
            
                #move the values from ro's comming positions 
                if np.sum(ro_ix[offsets]) == 0: 
                    #no common values in the left- and right- hand side indicies
                    self.idx[ro_ix] = self.idx[offsets]
                else:
                    #this operation is expensive but rare
                    ro_ix = np.flatnonzero( ro_ix ) #convert to indicies
                    intersect, ro_ind, offsets_ind = np.intersect1d(ro_ix, offsets, assume_unique=True, return_indices=True) 
                    self.idx[np.delete(ro_ix,ro_ind)] = self.idx[np.delete(offsets,offsets_ind)]

                #set ro and return the new ro
                self.idx[offsets] = ro
                if not self.forward: offsets = len(self.idx)-1-offsets%len(self.idx)
                return offsets

        def __iter__(self): return next(self)
        def __next__(self):
            i=-1
            while True: 
                i+=1
                if i >= len(self): return
                else:              yield i
    """        
    class ShuffleSampler(Sampler):
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, shuffle:bool=True): 
            super().__init__(length)
            self.idx, self.ix_last, self.shuffle = np.arange(length), length, shuffle
        def sampleOne(self)->int:
            ix = self.ix_last = self.ix_last - 1
            if ix == -1: 
                ix = self.ix_last = len(self.idx) - 1
                if self.shuffle:np.random.shuffle(self.idx)
            return self.idx[len(self.idx)-ix-1]

    class SampleOnLength(Sampler):
        def __init__(self, lengths, bins, sampler:Sampler):
            super().__init__(len(lengths))        
            self.bins, self.shuffle_in_buckets, self.sampler = bins, True, sampler

            n_buckets = len(bins)   
            buckets   = np.empty(n_buckets,dtype=object)
            for i in range(n_buckets): buckets[i]=[]
        
            lower,upper,step = bins[0],bins[-1],(bins[-1]-bins[0])/(n_buckets-1)
            #print(f"bins:{bins}\nn_buckets:{n_buckets}\nlower:{lower}, upper:{upper}, step:{step}")
            for i in range(len(lengths)): 
                length = lengths[i]-lower
                ib     = 0 if length<0 else n_bucket-1 if length>upper else int((length-lower)/step + .5) 
                buckets[ib].append(i)
        
            #fill empty buckets
            bucket_size = np.fromiter( ( len(b) for b in buckets ), dtype=np.int, count=n_buckets)
            ix_empty    = np.flatnonzero( bucket_size==0 )
            for i in ix_empty:
                #fill the empty bucket from the nearest left or right bucket
                i_n, stolen = 0, []
                while len(stolen) == 0:
                    i_n  += 1
                    ib    = i-i_n if i-i_n >= 0            and len(buckets[i-i_n]) > 1 else \
                            i+i_n if i+i_n  < len(buckets) and len(buckets[i+i_n]) > 1 else None
                    if ib is not None:
                        split = len(buckets[ib])//2
                        stolen.extend(buckets[ib][split : ])
                        buckets[ib] = buckets[ib][ : split]
                    buckets[i] = stolen
            
            #convert to arrays
            for i in range(n_buckets): buckets[i] = np.asarray(buckets[i],np.int64)
            self.buckets = buckets
            self.ix_last = np.fromiter( (len(b) for b in buckets), dtype=np.int64, count=len(buckets) )

        def sampleOne(self)->int:
            i_bucket = self.sampler()
            ix = self.ix_last[i_bucket] = self.ix_last[i_bucket] - 1        
            if ix == -1:
                ix = self.ix_last[i_bucket] = len(self.buckets[i_bucket])-1
                if self.shuffle_in_buckets: np.random.shuffle(self.buckets[i_bucket])
            return self.buckets[i_bucket][ix]


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
        if self.resampleLengths:
            lower, upper, step = np.min(self.lengths), np.max(self.lengths), 10
            if (upper-lower)//step < 10: step = 1
            bins         = np.arange( lower, upper+step, step)
            #print(f"lower:{lower}, upper:{upper}, step:{step}\nbins:{bins}")
            bins_sampler = random_bucket(nBuckets=len(bins))
            self.idx     = MyLanguageModelPreLoader.SampleOnLength(self.lengths, bins, sampler=bins_sampler)
            #print(f"self.idx.buckets:{self.idx.buckets}")
        else:
            self.idx   = LanguageModelPreLoader.ShuffleSampler(len(self.dataset.x.items), self.shuffle)

        self.batch = np.zeros((self.bs, self.bptt+1), dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,1:self.bptt+1] 
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.fromiter( (self.idx.sampleOne() for i in range(self.bs)), dtype=np.int64, count=self.bs)
        #self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        #self.ri    = np.zeros(self.bs, dtype=np.int)
        self.ri    = np.fromiter( (self.lengths[self.ro[i]] if self.backwards else 0  for i in range(self.bs)), 
                                  dtype=np.int64, count=self.bs)

        #t0 = time.perf_counter()
        """
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
        """

    def printJagged(self):
        for j in range(len(self.dataset.x.items)): print(f"r{j},{self.idx[j]} :{self.dataset.x.items[self.idx[j]]}")
    def on_epoch_begin(self, **kwargs):
        #after the first epoch get the direct location of ro in the source data 
        #ro_from = None if self.idx is None else  [self.idx[i] for i in self.ro]
        if self.idx is None: self.allocate_buffers()
        """            
        elif self.shuffle:
            ro_before = np.fromiter((self.idx[r] for r in self.ro), dtype=np.int, count=len(self.ro))
            self.ro = self.idx.shuffle(self.ro)
            ro_after  = np.fromiter((self.idx[r] for r in self.ro), dtype=np.int, count=len(self.ro))
            assert ((ro_after-ro_before)==0).all(), f"\nfailed   :{(ro_after-ro_before)}\nro_after :{ro_after}\nro_before:{ro_before}"
        self.idx.forward = not self.backwards 
        """            

         
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
            ro    = sampler.sampleOne() if ibuf else ro
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