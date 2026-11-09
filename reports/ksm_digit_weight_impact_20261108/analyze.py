"""Attribution to fixed stored E4-B weights. No model run or dataset change."""
import csv,json,re,sys,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.personalization.error_profile import normalize_for_syllable_alignment,align_syllables
from transformers import WhisperProcessor
from transformers.convert_slow_tokenizer import bytes_to_unicode
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    cache=ROOT/'outputs/personalization/error_profiles/predictions.csv'
    provenance=ROOT/'outputs/personalization/E4/KSM_M_45/E4-B/provenance.json'
    paths=[cache,provenance,ROOT/'src/personalization/e4_weighted.py',ROOT/'data/splits/personalization_split.csv']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with cache.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['speaker_id']=='KSM_M_45']
    assert len(rows)==168
    digit={i for i,r in enumerate(rows) if re.search('[0-9]',r['reference'])};assert len(digit)==6
    n=Counter();e=Counter();sd=Counter();de=Counter();eu=defaultdict(set);digit_e=Counter();digit_sd=Counter();digit_de=Counter()
    utterances=[]
    for i,r in enumerate(rows):
        ref=normalize_for_syllable_alignment(r['reference']);hyp=normalize_for_syllable_alignment(r['prediction']);n.update(ref)
        counts=Counter();effects=Counter()
        for a,b in align_syllables(ref,hyp):
            if a==b:continue
            kind='I' if a=='<EMPTY>' else 'D' if b=='<EMPTY>' else 'S';counts[kind]+=1
            if a!='<EMPTY>':
                e[a]+=1;eu[a].add(i)
                if kind=='S':sd[a]+=1
                else:de[a]+=1
                if i in digit:
                    digit_e[a]+=1;effects[a]+=1
                    if kind=='S':digit_sd[a]+=1
                    else:digit_de[a]+=1
        if i in digit:utterances.append(dict(index=i,audio=r['audio_member'],reference=r['reference'],prediction=r['prediction'],
            S=counts['S'],D=counts['D'],I=counts['I'],reference_syllable_errors=dict(effects)))
    p0=sum(e.values())/sum(n.values());saved=read(provenance)['syllable_weights']
    weights={};without={};details=[]
    for s in n:
        def weight(errors,error_utterances):
            rate=(errors+10*p0)/(n[s]+10)
            return 1+min(.5,max(0,rate-p0)) if n[s]>=5 and errors>=2 and error_utterances>=2 else 1.
        weights[s]=weight(e[s],len(eu[s]))
        # Descriptive counterfactual ONLY: keep N and global p0 fixed, subtract
        # six utterances' S/D events and their error-utterance support.
        without[s]=weight(e[s]-digit_e[s],len(eu[s]-digit))
        assert abs(weights[s]-saved[s])<1e-12
        if digit_e[s]:details.append(dict(syllable=s,N=n[s],total_errors=e[s],six_errors=digit_e[s],six_S=digit_sd[s],six_D=digit_de[s],
            total_error_utterances=len(eu[s]),six_error_utterances=len(eu[s]&digit),final_weight=weights[s],
            error_only_counterfactual_weight=without[s],direct_weight_delta=weights[s]-without[s],
            selected=weights[s]>1,gate_lost_without_six=weights[s]>1 and without[s]==1))
    tok=WhisperProcessor.from_pretrained('openai/whisper-small',language='korean',task='transcribe',local_files_only=True).tokenizer
    bd={v:k for k,v in bytes_to_unicode().items()};tokens=[];affected_syllables={s for s in n if weights[s]>without[s]+1e-12}
    sums=Counter();affected_utterances=set();weighted_utterances=set();inside_rows=Counter()
    for i,r in enumerate(rows):
        text=r['reference'].strip();ids=tok(text,add_special_tokens=False).input_ids
        chars=[];pos=0
        for ch in text:
            end=pos+len(ch.encode('utf-8'));chars.append((ch,pos,end));pos=end
        fragments=[bytes(bd[c] for c in tok.convert_ids_to_tokens(t)) for t in ids]
        assert b''.join(fragments)==text.encode('utf-8');pos=0
        for j,(token,b) in enumerate(zip(ids,fragments)):
            end=pos+len(b);overlap=[(s,min(end,y)-max(pos,x)) for s,x,y in chars if min(end,y)>max(pos,x)]
            w=1+sum(k*(weights.get(s,1)-1) for s,k in overlap)/len(b)
            cf=1+sum(k*(without.get(s,1)-1) for s,k in overlap)/len(b)
            changed=w-cf>1e-12;inc=w>1+1e-12
            sums.update(tokens=1,weighted=int(inc),affected=int(changed),extra_mass=w-1,direct_mass=w-cf,
                counterfactual_weighted=int(cf>1+1e-12))
            if inc:weighted_utterances.add(i)
            if changed:affected_utterances.add(i)
            if i in digit:
                inside_rows.update(tokens=1,weighted=int(inc),affected=int(changed),extra_mass=w-1,direct_mass=w-cf)
            is_digit=any(re.search('[0-9]',s) for s,k in overlap)
            if inc or changed or is_digit:tokens.append(dict(audio=r['audio_member'],is_six=i in digit,token_position=j,token_id=token,
                overlap=''.join(s for s,k in overlap),weight=w,error_only_counterfactual=cf,direct_delta=w-cf,contains_arabic_digit=is_digit))
            pos=end
    details.sort(key=lambda r:(-r['direct_weight_delta'],-r['six_errors']));table('syllable_contribution.csv',details);table('token_positions.csv',tokens)
    for r in utterances:
        r['S_D_used_by_selected_weights']=sum(c for s,c in r['reference_syllable_errors'].items() if weights[s]>1)
        r['weighted_syllable_errors']=json.dumps(r.pop('reference_syllable_errors'),ensure_ascii=False)
    table('six_utterances.csv',utterances)
    summary=dict(actual_p0=p0,N=sum(n.values()),S=sum(sd.values()),D=sum(de.values()),six_S=sum(digit_sd.values()),six_D=sum(digit_de.values()),six_I=sum(r['I'] for r in utterances),
        six_reference_error_types=len(digit_e),six_errors_on_selected_syllables=sum(v for s,v in digit_e.items() if weights[s]>1),
        contributed_selected_types=[r['syllable'] for r in details if r['selected']],
        directly_changed_weight_types=sorted(affected_syllables),gate_lost_types=[r['syllable'] for r in details if r['gate_lost_without_six']],
        total_selected_types=sum(w>1 for w in weights.values()),counterfactual_selected_types=sum(w>1 for w in without.values()),
        token_summary=dict(sums),affected_utterances=len(affected_utterances),weighted_utterances=len(weighted_utterances),inside_six=dict(inside_rows),
        digit_tokens=[r for r in tokens if r['contains_arabic_digit']],
        indirect_p0_delta_from_six=sum(digit_e.values())/sum(n.values()),
        diagnostic_definition='Hold all Train rows, observation N, and global p0 fixed; subtract six utterances S/D events and error-utterance support only. Not actual training or a new stored Profile.',
        caveat='Attribution to six utterances is not attribution to numerical notation errors alone; those utterances contain other ASR errors. Weight mass is not loss or gradient contribution.')
    write('summary.json',summary);write('input_checksums.json',before)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('SYLLABLES',json.dumps(details,ensure_ascii=False))

if __name__=='__main__':main()
