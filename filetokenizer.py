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
        self.tok_func,self.lang = tok_func,lang
        self.special_cases = special_cases if special_cases else defaults.text_spec_tok
        self.n_cpus  = ifnone(n_cpus, defaults.cpus)
        self.vocab   = vocab
        self.minToks = minToks
        self.tokPath = tokPath
        self.count   = 0
        self.dtype   = np.uint16 if len(self.vocab.itos) < pow(2,16)-1 else np.int32
        if self.special_cases: 
            tok = self.tok_func(self.lang)
            tok.add_special_cases(self.special_cases)

    def __repr__(self) -> str:
        res = f'{self.__name__} using {self.tok_func}:\n'
        return res

    def process_text(self, t:str, tok:BaseTokenizer) -> List[str]:
        "Process one text `t` with tokenizer `tok`."
        inPath = Path(t)
        if not inPath.exists(): 
            print(f"file does not exist{str(inPath)}")
            return ""
        else: 
            print(f"processing file:{inPath}")

        arrays  = []
        with inPath.open("r", encoding='utf-8') as f:
            for line in f:
                toks = tok.tokenizer(line)
                ids  = self.vocab.numericalize(toks) 
                
                if len(toks) < self.minToks: continue
                arrays.append( np.asarray(ids, dtype=self.dtype) )
        
        if len(arrays)>0:
            pathIds = self.tokPath/(inPath.stem+"-ids.npy")
            pathIds.parent.mkdir(parents=True,exist_ok=True)
            with pathIds.open("wb") as f:
                np.save(f, np.asarray(arrays), allow_pickle=True, fix_imports=False)           
        return t
 
    def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts` in one process."
        tok = self.tok_func(self.lang)
        return [self.process_text(t, tok) for t in texts]

    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1: return self._process_all_1(texts)
        print(f"FileTokenizer.process_all files:{len(texts)} on n_cpus:{self.n_cpus} and minToks:{self.minToks}")
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])
        
    def getIds(self):
        files = list(self.tokPath.glob("*-ids.npy"))
        idArrays=[]
        for fp in files:
            with fp.open("rb") as f:
                a = np.load(f)
                if len(a) > 0: idArrays.extend( a )
        return idArrays 
        
class FileTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, mark_fields:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields
    def process_one(self, item):  return self.tokenizer._process_all_1([item])[0]
    def process(self, ds): self.tokenizer.process_all(ds.items)
