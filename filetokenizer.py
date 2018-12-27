from fastai import * 
from fastai.basics import * 
from fastai.text import * 
import sentencepiece as spm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import fastai.text.transform

from fastai_sentencepiece import *

class FileTokenizer():
    "Put together rules and a tokenizer function to tokenize text with multiprocessing."
    def __init__(self, tokPath:Path, tok_func:Callable, lang:str, vocab:fastai.text.transform.Vocab, 
                 special_cases:Collection[str]=None, n_cpus:int=None, minToks:int=5):
        self.tok_func,self.lang,self.special_cases = tok_func,lang,special_cases
        self.pre_rules  = text.transform.defaults.text_pre_rules
        self.pre_rules.append(rm_extra_lineshift)
        self.post_rules = text.transform.defaults.text_post_rules
        self.special_cases = special_cases if special_cases else defaults.text_spec_tok
        self.n_cpus = ifnone(n_cpus, defaults.cpus)
        self.vocab  = vocab
        self.minToks = minToks
        self.tokPath = tokPath
        
        self.count=0

    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_func.__name__} in {self.lang} with the following rules:\n'
        for rule in self.pre_rules: res += f' - {rule.__name__}\n'
        for rule in self.post_rules: res += f' - {rule.__name__}\n'
        return res

    def process_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Process one text `t` with tokenizer `tok`."
        inPath = Path(t)
        if not inPath.exists(): 
            print(f"file does not exist{str(inPath)}")
            return ""
        
        pathIds = self.tokPath/(inPath.stem+"-ids.npy")
        pathIds.parent.mkdir(parents=True,exist_ok=True)
        dtype =  np.int16 if len(vocab.itos) < 2^15/2-1 else np.int32  #should test whether we can use uint16
        arrays = []
        with inPath.open("r") as f:
            for line in f:
                for rule in self.pre_rules: line = rule(line)
                toks = tok.tokenizer(line)
                for rule in self.post_rules: toks = rule(toks)
                ids = self.vocab.numericalize(toks) 
                
                if len(toks) < self.minToks: continue
                arrays.append( np.asarray(ids, dtype=dtype) )
        
        if len(arrays)>0:
            with pathIds.open("wb") as f:
                np.save(f, np.asarray(arrays), allow_pickle=True, fix_imports=False)
                
        return t
 
    def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts` in one process."
        tok = self.tok_func(self.lang)
        if self.special_cases: tok.add_special_cases(self.special_cases)
        return [self.process_text(t, tok) for t in texts]

    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1: return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])
        
    @staticmethod
    def getIds_from_file(files):
        idArrays=[]
        for fp in files:
            with fp.open("rb") as f:
                a = np.load(f)
                if len(a) > 0: idArrays.extend( a )
        return idArrays 
    
    #def getIds(self, n_cpus=defaults.cpus):
    def getIds(self, n_cpus=1):
        #threading does not help on speed in this case :(
        files = list(self.tokPath.glob("*-ids.npy"))
        
        #3use_cores = max(1,defaults.cpus)
        print(f"threading with on {n_cpus} cores")
        
        idArrays = self.getIds_from_file(files)
        #pool = ThreadPool(n_cpus) 
        #results = pool.map(FileTokenizer.getIds_from_file, partition_by_cores(files, n_cpus))
        #pool.close() 
        #pool.join()
        #idArrays=[]
        #for a in results:idArrays.extend(a)
        #idArrays = np.asarray(idArrays,dtype=object)
        
        return idArrays 
        
class FileTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, mark_fields:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields

    def process_one(self, item):  return self.tokenizer._process_all_1([item])[0]
    def process(self, ds):
        print("FileTokenizeProcessor process")
        #ds.items = _join_texts(ds.items, self.mark_fields)
        self.tokenizer.process_all(ds.items)
        #ds.items = self.tokenizer.process_all(ds.items)
        #ds.items = tokens
