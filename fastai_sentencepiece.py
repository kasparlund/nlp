from fastai.text import * 
from pathlib import *
from functools import partial
import re
import shutil
 
import sentencepiece as spm

def rm_extra_lineshift(t:str) -> str:
    return re.sub('[\r\n]+', '\n', t)

class SentencepieceWikiModel:
    def __init__(self, lang:str, pathJson:Path, pathcsv:Path, pathTxt:Path, pathVocab:Path, minWords, 
                 vocab_size:int=32000, model_type:str='unigram', 
                 rules=text.transform.defaults.text_pre_rules, chunksize=500000 ):  #should include removal of repetitions
        self.lang           = lang
        self.pathJson       = pathJson
        self.pathVocab      = pathVocab
        self.pathcsv        = pathcsv
        self.pathTxt        = pathTxt
        self.vocab_size     = vocab_size
        self.model_type     = model_type
        self.rules          = rules
        self.rules.append(rm_extra_lineshift)
        self.minWords       = minWords
        self.chunksize      = chunksize

    def wikijson2TrainingData(self):
        #Generate text files for training af sentencepiece vocabulary and a csv-file for training a languagemodel with the vocabulary"
        
        def save_sections( data, f_out, fileCount ):
            if not self.pathTxt.exists(): self.pathTxt.mkdir(parents=True, exist_ok=True)
            p = self.pathTxt / f"{fileCount}.txt"
            with p.open("w+", encoding="utf-8") as fw:
                fw.write( "\n".join(data["text"]) )
            data.to_csv(f_out, index=False, header=False, mode='a', encoding='utf-8')
        
        pathParts = len(self.pathJson.parts)        

        if not self.pathcsv.parent.exists(): self.pathcsv.parent.mkdir(parents=True, exist_ok=True)
        with self.pathcsv.open("w", encoding='utf-8') as f_out:
            fileCount = 0
            fields    = ['text', 'wordcount']
            data      = pd.DataFrame(columns=fields)
            data.to_csv(f_out, index=False, mode='a', )
            
            sections = []
            for fn in self.pathJson.glob("**/wiki*"):
                nbWords=0
                with open(fn, encoding='utf-8') as f:
                    for line in f:
                        section = json.loads(line)
                        
                        if section['text'].find(section['title']) >=0 : section['text'] = section['text'][:len(section['title']):]
                
                        section['text']      = reduce(lambda t, rule: rule(t), self.rules, section['text'])
                        section['wordcount'] = len(re.findall(r'\w+',section['text']))

                        if section["wordcount"] > self.minWords:
                            sections.append({k:section[k] for k in fields})
                            nbWords += section["wordcount"]
                
                #if len(sections) > self.chunksize: 
                if nbWords > self.chunksize: 
                    data = pd.DataFrame(sections)
                    sections = []
                    
                    save_sections(data, f_out, fileCount)
                    fileCount += 1
                    data = None
                    
            if len(sections) > 0: 
                data = pd.DataFrame(sections)
                sections = []
                    
                save_sections(data, f_out, fileCount)
                fileCount += 1
                data = None
                
            
    def getUserdefinedSymbols(self): 
        return defaults.text_spec_tok
        """
                [text.transform.BOS,
                text.transform.PAD,
                text.transform.TK_MAJ,
                text.transform.TK_UP,
                text.transform.TK_REP,
                text.transform.TK_WREP,
                text.transform.FLD ] 
        """

    def trainVocabulary(self): 
        if self.pathVocab.exists():
            self.pathVocab.mkdir(parents=True, exist_ok=True)
        model_prefix = self.pathVocab / "m"
    
        #Set the following controls to sentencepiece values until there is a release where we can set the token value
        #Note taku910 has already made the change but the pip of sentencepiewce version has not been updated 
        text.transform.UNK = "<unk>"
        #text.transform.BOS = "<s>"
        #text.transform.PAD = "<pad>"
    
        #create control ids for the rest of the fastai control tokens in case the user needs them
        #it is the responsibility of fastai to generate and use the control tokens them and apply them before decoding
        #Fx applying TK_MAJ after tokenization would change She to two token TK_MAJ+she.
        #Problem! Sentencepiece would tokenize "Elle" as _Elle so our deal_caps would not catch it
        str_specialcases = ",".join(self.getUserdefinedSymbols()) 
    
        pathSrc_list = [str(s) for s in self.pathTxt.glob("**/*.txt")]
        pathSrc_list= ",".join(pathSrc_list)
        #we enble all control symbols so that the tokenizations cleans the input text
        #but we also create them as userdefined symbols in order to allocate and id for each
        #f"--bos_id=-1 " \
        #f"--eos_id=-1 " \
        #f"--pad_id=-1 " \
        sp_params = f"--input={pathSrc_list} "  \
                    f"--unk_piece={UNK} " \
                    f"--bos_piece={BOS} " \
                    f"--eos_piece=xxeos " \
                    f"--pas_piece={PAD} " \
                    f"--user_defined_symbols={str_specialcases} " \
                    f"--character_coverage=1.0 " \
                    f"--max_sentence_length=4096 " \
                    f"--input_sentence_size={int(1e7)} " \
                    f"--shuffle_input_sentence=true " \
                    f"--model_prefix={model_prefix} " \
                    f"--vocab_size={self.vocab_size} " \
                    f"--model_type={self.model_type} " 
    
        #f"--split_by_number=1 " \
        #hard_vocab_limit=False
        #use_all_vocab
        #print(sp_params)
        spm.SentencePieceTrainer.Train(sp_params)
        
        #convert sentencepieces vocabulary to a format fastai can read
        with open( self.pathVocab/"m.vocab", 'r') as f:
            vocab = [line.split('\t')[0] for line in f.readlines()]
        with open( self.pathVocab / "itos.pkl", "wb") as f:    
            pickle.dump(vocab, f)


class SentencepieceTokenizer(BaseTokenizer):
    def __init__(self, lang:str, pathVocab:Path):
        self.pathVocab = pathVocab
        self.vocab_    = Vocab(pickle.load(open(self.pathVocab/'itos.pkl', 'rb')))
        self.tok       = spm.SentencePieceProcessor()
        
        self.tok.Load(str(pathVocab / 'm.model'))
        text.transform.UNK = "<unk>"

    @staticmethod    
    def create(lang:str, pathVocab:Path):
        return SentencepieceTokenizer(lang, pathVocab)

    def tokenizer(self, t:str) -> List[str]:
        return self.tok.EncodeAsPieces(t)
    
    def add_special_cases(self, toks:Collection[str]):
        #this should have been done when training sentencepiece
        pass
    
    def vocab(self): return self.vocab_
    