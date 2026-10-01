"""NLLB-200-distilled-600M translation with a persistent cache and back-translation chrF filter (B3).

Cache files are JSONL, one record per (dataset item, language):
  {"item_id", "lang", "question", "choices", "back_question", "chrf"}
WMDP translations are hazardous-adjacent and go to $DSG_PRIVATE/translations; MMLU ones to
$DSG_CACHE/translations.
"""
import json
from pathlib import Path

from dsgx import paths

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "fr": "fra_Latn", "es": "spa_Latn",
         "zh": "zho_Hans", "ar": "arb_Arab", "ru": "rus_Cyrl"}
SRC = "eng_Latn"


def cache_file(dataset: str, lang: str) -> Path:
    root = paths.private_dir() if dataset.startswith("wmdp") else paths.cache_dir()
    p = root / "translations" / lang / f"{dataset}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_cache(dataset: str, lang: str) -> dict:
    p = cache_file(dataset, lang)
    out = {}
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


class Translator:
    def __init__(self, device="cuda", batch_size=16, max_len=400):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
        self.device, self.bs, self.max_len = device, batch_size, max_len

    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]:
        import torch

        out = []
        self.tok.src_lang = src
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                                          max_new_tokens=self.max_len, num_beams=1)
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out


def chrf(hyp: str, ref: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_chrf(hyp, [ref]).score)


def translate_items(items, lang: str, translator: Translator, progress=None) -> int:
    """Translate (question + 4 choices) of MCQ items into `lang`, back-translate the question,
    append to the cache. Returns the number of new records."""
    tgt = LANGS[lang]
    if not items:
        return 0
    dataset = items[0].dataset
    have = load_cache(dataset, lang)
    todo = [it for it in items if it.item_id not in have]
    new = 0
    with cache_file(dataset, lang).open("a") as f:
        for s in range(0, len(todo), 32):
            chunk = todo[s:s + 32]
            segs = [x for it in chunk for x in (it.question, *it.choices)]
            tr = translator(segs, SRC, tgt)
            qs = [tr[5 * j] for j in range(len(chunk))]
            back = translator(qs, tgt, SRC)
            for j, it in enumerate(chunk):
                rec = {"item_id": it.item_id, "lang": lang, "question": tr[5 * j],
                       "choices": tr[5 * j + 1:5 * j + 5], "back_question": back[j],
                       "chrf": chrf(back[j], it.question)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new += 1
            f.flush()
            if progress:
                progress(len(chunk))
    return new
