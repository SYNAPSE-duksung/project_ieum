"""Explicit Korean scoring rules; corpus scores use summed edit counts."""
from collections import defaultdict
import unicodedata
import jiwer

NORMALIZATION = 'NFC; lowercase; Unicode punctuation -> space; collapse whitespace; CER removes whitespace; WER uses whitespace words; numbers unchanged'


def normalize(text):
    text = unicodedata.normalize('NFC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def counts(reference, prediction, normalized=True):
    ref = normalize(reference) if normalized else unicodedata.normalize('NFC', reference).strip()
    hyp = normalize(prediction) if normalized else unicodedata.normalize('NFC', prediction).strip()
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(''.join(ref.split()), ''.join(hyp.split()))
    result = {}
    for prefix, obj in [('word', words), ('char', chars)]:
        result[prefix + '_errors'] = obj.substitutions + obj.deletions + obj.insertions
        result[prefix + '_reference'] = obj.hits + obj.substitutions + obj.deletions
    result['cer'] = result['char_errors'] / result['char_reference'] if result['char_reference'] else None
    result['wer'] = result['word_errors'] / result['word_reference'] if result['word_reference'] else None
    return result


def aggregate(records):
    def summarize(rows):
        good = [r for r in rows if r['status'] == 'success']
        out = dict(attempted=len(rows), successful=len(good), failed=len(rows)-len(good))
        for mode in ('normalized', 'raw'):
            total = {key: sum(r['scores'][mode][key] for r in good)
                     for key in ('char_errors', 'char_reference', 'word_errors', 'word_reference')}
            total['cer'] = total['char_errors'] / total['char_reference'] if total['char_reference'] else None
            total['wer'] = total['word_errors'] / total['word_reference'] if total['word_reference'] else None
            out[mode] = total
        out['duration_sec'] = sum(float(r['duration_sec']) for r in good)
        out['inference_sec'] = sum(r['inference_sec'] for r in good)
        out['loading_sec'] = sum(r['loading_sec'] for r in good)
        out['peak_allocated_mib'] = max((r['peak_allocated_mib'] for r in good), default=0)
        out['peak_reserved_mib'] = max((r['peak_reserved_mib'] for r in good), default=0)
        return out
    result = dict(normalization=NORMALIZATION, overall=summarize(records))
    for field in ('speaker_id', 'disability_group'):
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row)
        result['by_' + field] = {key: summarize(rows) for key, rows in groups.items()}
    return result
