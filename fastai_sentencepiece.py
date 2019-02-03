<<<<<<< HEAD
from fastai.text import * 
from pathlib import *
from functools import partial
import re
import string
import shutil
 
import sentencepiece as spm

#def rm_escaped_quotes(t:str) -> str: return t.replace("\\'","'").replace('\\"','"')
def rm_empty_quotes(t:str) -> str:     
    return re.sub("[\"'][\"']+","",t)

def rm_empty_lists(t:str) -> str:
    t= re.sub("\s*[,;][\s,;]+", ", ", t) # replace  ", ,, ;" or ",, ," or  ", ," with ","
    t= re.sub("\([\s,;]","(", t) # replace (, or ( , with (
    t= re.sub("[\s,;]\)",")", t) # replace ,) or ,)  with )
    #t= re.sub("\([\s,;]+","(", t) # replace (, or ( , with (
    #t= re.sub("[\s,;]+\)",")", t) # replace ,) or ,)  with )
    t= re.sub("\([^\d\w\r\n]+\)", "", t)     # remove  () ( ) (  )  (, ) (, )  (,,, ; ) but leave any parantese with a letter or number
    return t

def lower(t:str) -> str: return t.lower()
def trim(t:str) -> str:  return t.strip()
def extract_link_title(t:str) -> str: 
    "the below evaluation order is important"
    def last_pip(m):
        gt = m.group(1)
        last = gt.rfind('|')
        return gt if last==-1 else gt[last+1:]

    t=re.sub('\[\[(?:[^:]*:[^\]]*)\]\]', '', t)  #remove [[file:blabala]], [[image:blabala]] etc
    t=re.sub('\[\[([^\]\[:]+)\]\]', last_pip, t) #replace [[samething|othering]] or [[otherthing]] with otherthing
    return re.sub('(?:[\]\[])', "", t)           #remove the remaning [[, [, ], ]] 
def rm_stray_tags(t:str) -> str: 
    #return re.sub("<[/\w\s]*>", "", t)           # remove  (<ref> </ref> </ ref> < /ref > <ref /> <br> <nowiki> < / nowiki> < / > etc
    return re.sub("<(style|script)[^<>]*>.*?</\1>|</?[a-z][a-z0-9]*[^<>]*>|<!--.*?-->","",t) # remove any remaining html tags
def spec_add_more_spaces(t:str) -> str:
    "replace this nonbreakling charater with space"
    return t.replace("\xa0","")
    #return re.sub(r'([,;%°\*\+-_:\.\(\)/#$§£€•<>]\?)', r' \1 ', t)
def my_replace_wrep(t:str) -> str:
    def _replace_wrep(m:Collection[str]) -> str:
        c,_ = m.groups()
        n   = len(m.string[m.start():m.end()].split(c))
        return f' {TK_WREP}{n-1}{c} '
    re_rep = re.compile(r'\b(\w+)\s+(\1\b\s*)+')
    return re_rep.sub(_replace_wrep, t)
                   
def my_replace_rep(t:str) -> str:
    "Replace repetitions at the character level in `t`."
    def _replace_rep(m:Collection[str]) -> str:
        c,cc= m.groups()
        return f' {TK_REP}{len(cc)+1}{c} '
    re_rep = re.compile(r'(\S)(\1\1\1+)')
    return re_rep.sub(_replace_rep, t)


def count_alphas(t:str) -> str: 
    len_spaces       = t.count(" ")
    len_alphas       = len(re.findall('[A-Za-z\"“”‘’]', t)) 
    len_list_symbols = len(re.findall('[,;:•—|\\/]', t))
    if len_spaces==0 : len_spaces=1e-6
    return len_alphas/len(t), len_alphas/len_spaces, len_list_symbols/len_spaces  #alpha ratio, alphas/word, tech symbols/word
    
#not used at present: replace_rep, replace_wrep
spm_rules = [fix_html, lower, extract_link_title, rm_stray_tags, rm_empty_lists, rm_empty_quotes,
             my_replace_rep, my_replace_wrep, spec_add_more_spaces, rm_useless_spaces, trim ]
class SentencepieceWikiVocab:
    def __init__(self, lang:str, pathJson:Path, pathcsv:Path, pathTxt:Path, pathVocab:Path, vocab_size:int=4000, model_type:str='unigram'):
        self.lang           = lang
        self.pathJson       = pathJson
        self.pathVocab      = pathVocab
        self.pathcsv        = pathcsv
        self.pathTxt        = pathTxt
        self.vocab_size     = vocab_size
        self.model_type     = model_type

    def wikijson2TrainingData(self, rules=spm_rules, min_alpha=0.66, max_sep_pr_space=0.17, 
                                    max_characters_pr_line=6000, min_words_pr_line=6, chunksize=int(4e7)):
        #Generate text files for training af sentencepiece vocabulary and a csv-file for training a languagemodel with the vocabulary"
        
        def save_sections( data, f_out, fileCount, header=False):
            if not self.pathTxt.exists(): self.pathTxt.mkdir(parents=True, exist_ok=True)
            p = self.pathTxt / f"{fileCount}.txt"
            with p.open("w+", encoding="utf-8") as fw:
                fw.write( "\n".join(data["text"]) )
            data.to_csv(f_out, index=False, header=header, mode='a', encoding='utf-8')
        
        pathParts = len(self.pathJson.parts)        

        if not self.pathcsv.parent.exists(): self.pathcsv.parent.mkdir(parents=True, exist_ok=True)
        i_csv_write=0
        with self.pathcsv.open("w", encoding='utf-8') as f_out:
            fileCount = 0
            fields    = ["text", "wordcount", "lettercount", "linecount", "words_pr_line", "letters_pr_line", "letters_pr_word", "alpha_r1", "alpha_r2", "alpha_r3"]
            data      = pd.DataFrame(columns=fields)
            #data.to_csv(f_out, index=False, mode='a', encoding='utf-8' )
            
            sections = []
            nbWords  = 0
            for fn in self.pathJson.glob("**/wiki*"):
                with open(fn, encoding='utf-8') as f:
                    for line in f:
                        section = json.loads(line)
                        
                        if section['text'].find(section['title']) >=0 : section['text'] = section['text'][len(section['title']):]

                        #first rough cleanup
                        text = section['text']
                        text = reduce(lambda t, rule: rule(t), rules, text)
                        txt_selected = []
                        for tl in text.splitlines() :
                            tln         = len(tl)
                            if tln==0: continue
                            wpl          = len(re.findall(r' ',tl))
                            alpha_pr_sentence, alpha_pr_word, sep_pr_space = count_alphas(tl)
                            if tln < max_characters_pr_line and wpl >= min_words_pr_line and \
                               alpha_pr_sentence >= min_alpha and sep_pr_space <= max_sep_pr_space:
                                txt_selected.append(tl)

                        text = "\n".join(txt_selected) if len(txt_selected)>0 else ""
                        
                        section['text'] = text
                        if len(txt_selected) > 0 and len(text)>0:
                            section['wordcount']       = len(re.findall(r' ',text))
                            section['lettercount']     = len(text)
                            section['linecount']       = len(txt_selected)
                            section['words_pr_line']   = section['wordcount']   / (section['linecount']) #+1 because we applied trim
                            section['letters_pr_line'] = section['lettercount'] / (section['linecount']) #+1 because we applied trim
                            section['letters_pr_word'] = section['lettercount'] / (section['wordcount']+1) #+1 to avoid div by zero
                            alphas = count_alphas(text)
                            section['alpha_r1'] = alphas[0]
                            section['alpha_r2'] = alphas[1]
                            section['alpha_r3'] = alphas[2]

                        if len(text) > 0:
                            sections.append({k:section[k] for k in fields})
                            nbWords += section["wordcount"]

                        if nbWords > chunksize: 
                            data = pd.DataFrame(sections)
                            sections = []
                            nbWords = 0
                            save_sections(data, f_out, fileCount, header=i_csv_write==0)
                            i_csv_write+=1
                            fileCount += 1
                            data = None
                    
            if len(sections) > 0: 
                data = pd.DataFrame(sections)
                sections = []
                    
                save_sections(data, f_out, fileCount)
                fileCount += 1
                data = None

    @staticmethod            
    def getControlSymbols(): 
        return  [text.transform.UNK, 
                text.transform.BOS,
                "xxeos",
                text.transform.PAD]

    @staticmethod        
    def getUserdefinedSymbols(): 
        return  [text.transform.FLD, 
                text.transform.TK_MAJ,
                text.transform.TK_UP,
                text.transform.TK_REP,
                text.transform.TK_WREP,
                "0","1","2","3","4","5","6","7","8","9",
                "°","%","$","§","£","€",
                "(",")","<",">","\"","\'","“","”","‘","’","!",
                ",",";",":",".","•","—","|","\\","/",
                "*","+","-","=","⁄","′","_","#","&","?"]

    def createParameters(self, sub_iterations:int): 
        model_prefix = self.pathVocab / "m"
    
        #Set the following controls to sentencepiece values until there is a release where we can set the token value
        #Note taku910 has already made the change but the pip of sentencepiewce version has not been updated 
    
        #create control ids for the rest of the fastai control tokens in case the user needs them
        #it is the responsibility of fastai to generate and use the control tokens them and apply them before decoding
        #Fx applying TK_MAJ after tokenization would change She to two token TK_MAJ+she.
        #Problem! Sentencepiece would tokenize "Elle" as _Elle so our deal_caps would not catch it
        str_specialcases = ",".join(SentencepieceWikiVocab.getUserdefinedSymbols()) 
    
        pathSrc_list = [str(s) for s in self.pathTxt.glob("**/*.txt")]
        pathSrc_list= ",".join(pathSrc_list)
        #we enble all control symbols so that the tokenizations cleans the input text
        #but we also create them as userdefined symbols in order to allocate and id for each
        #           f"--num_threads={defaults.cpus} " \         
        #           f"--num_sub_iterations={sub_iterations} " 
        sp_params = f"--input={pathSrc_list} "  \
                    f"--shuffle_input_sentence=true " \
                    f"--input_sentence_size={int(1.8e6)} " \
                    f"--max_sentence_length=4096 " \
                    f"--unk_id=0 " \
                    f"--bos_id=1 " \
                    f"--eos_id=2 " \
                    f"--pad_id=3 " \
                    f"--unk_piece={text.transform.UNK} " \
                    f"--bos_piece={text.transform.BOS} " \
                    f"--eos_piece=xxeos " \
                    f"--pad_piece={text.transform.PAD} " \
                    f"--user_defined_symbols={str_specialcases} " \
                    f"--split_by_number=true " \
                    f"--character_coverage=0.99988 " \
                    f"--model_prefix={model_prefix} " \
                    f"--vocab_size={self.vocab_size} " \
                    f"--model_type={self.model_type} " 
                    
        return sp_params, model_prefix

    def trainVocabulary(self, sub_iterations:int=4) : 
        sp_params, model_prefix = self.createParameters(sub_iterations)
        print(f"running spm.SentencePieceTrainer.Train(sp_params) with sp_params:\n{sp_params}")
        if self.pathVocab.exists(): self.pathVocab.mkdir(parents=True, exist_ok=True)
        if not model_prefix.parent.exists(): model_prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.Train(sp_params)
        print("finised training sentencepiece")

    def convertSPVocab2FastaiVocab(self):   
        "convert sentencepieces vocabulary to a format fastai can read"
        with (self.pathVocab/"m.vocab").open('r', encoding='utf-8') as f:
            vocab = [line.split('\t')[0] for line in f.readlines()]
        with (self.pathVocab / "itos.pkl").open( "wb") as f:    
            pickle.dump(vocab, f)

class SentencepieceTokenizer(BaseTokenizer):
    
    class SentencepieceVocab(Vocab):
        "Use sentencepiece to numericalize and textify."
        def __init__(self, itos:Collection[str], spp:spm.SentencePieceProcessor):
            super().__init__(itos)
            self.spp = spp

        def textify(self, nums:Collection[int], sep=' ') -> List[str]:
            "Convert a list of `nums` to their tokens."
            return spp.DecodeIds(nums)

    def __init__(self, lang:str, pathVocab:Path):
        self.lang       = lang
        self.pathVocab  = pathVocab
        self.tok        = spm.SentencePieceProcessor()
        self.tok.Load(str(pathVocab / 'm.model'))
        self.vocab_     = SentencepieceTokenizer.SentencepieceVocab(pickle.load(open(self.pathVocab/'itos.pkl', 'rb')),self.tok)
        self.pre_rules  = spm_rules
        self.post_rules = []

    def __repr__(self) -> str:
        res = f'Tokenizer {self.__name__} unigram model in {self.lang} with the following rules:\n'
        for rule in self.pre_rules: res  += f' - {rule.__name__}\n'
        for rule in self.post_rules: res += f' - {rule.__name__}\n'

    @staticmethod    
    def create(lang:str, pathVocab:Path):
        return SentencepieceTokenizer(lang, pathVocab)

    def tokenizer(self, t:str) -> List[str]:
        for rule in self.pre_rules: t = rule(t)
        toks = self.tok.EncodeAsPieces(t)
        for rule in self.post_rules: toks = rule(toks)
        #insert this to signal the beginning of a new sentence
        toks.insert(0,text.transform.BOS)
        toks.append("xxeos")
        return toks
    
    def add_special_cases(self, toks:Collection[str]):
        #this is not necessay with sentencepiece unigram model 
        pass
    
    def vocab(self): return self.vocab_
