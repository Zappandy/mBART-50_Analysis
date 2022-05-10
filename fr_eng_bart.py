from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_metric
from unidecode import unidecode
import nltk

french_sentence = "Reprise de la session"
english_sentence = "Resumption of the session"
print(french_sentence)
print(english_sentence)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
fr_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fr_XX")
eng_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")  # to have the first char correspond with the language code.


src_file = "europarl-v7.fr-en.fr"
trg_file = "europarl-v7.fr-en.en"

def tokenize_corpus(corpus_file, n=2, tokenizer=fr_tokenizer):
    """
    :param n: number of sentences to fetch
    """
    with open(corpus_file, 'r', encoding="utf-8") as f:
        corpus = f.readlines()
    #return [tokenizer(line, return_tensors="pt") for line in corpus[:n]]
    return [tokenizer(unidecode(line), return_tensors="pt") for line in corpus[:n]]

def translate(sentences, trg_lang_code="en_XX"):
    """
    : param sentences: corpus containing tokenized sentences
    """
    translated_sents = list()
    for sent in sentences:
        generated_tokens = model.generate(**sent, num_beams=1, max_length=512, # is mx 512 or 1024?
                    forced_bos_token_id=fr_tokenizer.lang_code_to_id[trg_lang_code])
        decoded_tokens = fr_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_sents.append(decoded_tokens)
    return translated_sents

def idtensor_to_tokens(tensor, tokenizer):
    flattened = tensor.squeeze().detach().numpy()
    return tokenizer.convert_ids_to_tokens(flattened)

def iterate_tensors(tokenized_sents, tokenizer=fr_tokenizer):
    return [idtensor_to_tokens(t["input_ids"], tokenizer) for t in tokenized_sents]
source_sents = tokenize_corpus(src_file)
src_sents_from_ids = iterate_tensors(source_sents)
target_sents = tokenize_corpus(trg_file, tokenizer=eng_tokenizer)
eval_sents_from_ids = iterate_tensors(target_sents)

translations = translate(source_sents)
hypotheses = [eng_tokenizer(sent, return_tensors="pt") for sent in translations]  # do I need the tensors?
hypotheses = iterate_tensors(hypotheses)
references = eval_sents_from_ids
#hypothesis = hypotheses[0]["input_ids"][0]  # possible flattening if not using tensors
print(references)
print(hypotheses)
smoothing = nltk.translate.bleu_score.SmoothingFunction().method1
bleu = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, smoothing_function=smoothing)  # hypothesis should be the google translate. reference replaced by example
print(bleu)
# 8 questions for exam: 50 to 100 answer
