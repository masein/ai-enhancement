import re, json
NUM = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')
def _norm(s): return re.sub(r'\bsept\b','sep',s.replace('’',"'").replace('½',' 1/2'),flags=re.I)
def _ampm(x):
    # "7-11 am" / "2–3pm" / "9:30–11:30 am" read as "7 am-11 am", so a range still says each time
    x = re.sub(r'(?<![\d:])(\d{1,2}(?::\d\d)?)\s*(?:[-–—]|\bto\b|\buntil\b|\btill\b)\s*(\d{1,2}(?::\d\d)?)\s*(am|pm|a\.m\.|p\.m\.)(?!\w)', r'\1 \3-\2 \3', x, flags=re.I)
    return re.sub(r'(\d)\s*(am|pm|a\.m\.|p\.m\.)(?!\w)', r'\1 \2', x, flags=re.I)
def _has(text, v, cs=False):
    text, v = _ampm(text), _ampm(v)
    t, v = (text, v) if cs else (text.lower(), v.lower())
    if re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)', t): return True
    # "9:30 am" also matches a bare "9:30" that isn't marked pm (and the other way round)
    m=re.fullmatch(r'(\d{1,2}:\d\d)\s*(am|pm)', v)
    if m and re.search(r'(?<![\d:])'+re.escape(m.group(1))+r'(?!\d)(?!\s*(?:am|pm|a\.m|p\.m))', t): return True
    return False
SIGNOFF=re.compile(r"(let me know|hope (this|that) helps|feel free|would you like|if you need|happy to help|anything else|want me to|shall i)",re.I)
def _lines(a):
    # the answer's own lines: no code fences, no lead-in ending in ":", no closing offer ("Let me know if…");
    # when the answer puts its result in a code block, the lines inside that block are the answer
    fence=re.search(r'```[^\n]*\n(.*?)```', a, re.S)
    if fence and fence.group(1).strip(): a=fence.group(1)
    ls=[l for l in a.splitlines() if l.strip() and not l.strip().startswith('```')]
    if len(ls)>1 and ls[0].rstrip().endswith(':'): ls=ls[1:]
    if len(ls)>1 and SIGNOFF.search(ls[-1]) and len(ls[-1].split())<=20: ls=ls[:-1]
    return ls
def _body(a): return '\n'.join(_lines(a))
_ONES={w:i for i,w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split())}
_TENS={w:10*i for i,w in enumerate('_ _ twenty thirty forty fifty sixty seventy eighty ninety'.split()) if w!='_'}
TIME=re.compile(r'(?<![\d:.])(\d{1,2})(?::(\d{2})|\.(\d{2})(?=\s*(?:am|pm|a\.m\.|p\.m\.))|(?=\s*(?:am|pm|a\.m\.|p\.m\.)))\s*(am|pm|a\.m\.|p\.m\.)?(?![a-z\d])', re.I)
def _hm(m):
    # (hour on a 24-hour clock, minute, whether am/pm was given) from a TIME match
    h,mi,ap=int(m.group(1)),int(m.group(2) or m.group(3) or 0),(m.group(4) or '').lower()[:1]
    if ap=='p' and h<12: h+=12
    if ap=='a' and h==12: h=0
    return h,mi,bool(ap)
def clock_times(text):
    out=set(); t=_ampm(text)
    t=re.sub(r'\b(noon|midday)\b','12:00 pm',t,flags=re.I); t=re.sub(r'\bmidnight\b','12:00 am',t,flags=re.I)
    hours={w:i for w,i in _ONES.items() if 1<=i<=12}
    t=re.sub(r'\b(at|by|from|until|till|before|after)\s+(%s)\b(?!\s*(?:hours?|minutes?|mins?|people|days?|weeks?))'%'|'.join(hours), lambda m:m.group(1)+' '+str(hours[m.group(2).lower()])+':00', t, flags=re.I)
    t=re.sub(r'\b(at|by|from|until|till|before|after)\s+(\d{1,2})\b(?![:%\d]|[.,]\d|\s*(?:am|pm|a\.m|p\.m|st|nd|rd|th|hours?|minutes?|mins?|people|days?|weeks?|percent|%|kg|km|[a-z]+\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)))', lambda m:m.group(1)+' '+m.group(2)+':00', t, flags=re.I)
    for m in TIME.finditer(t):
        h,mi,ap=_hm(m)
        out.add((h,mi))
        if not ap and h<=12: out.add(((h+12)%24,mi))
    return out
def word_numbers(text):
    # numbers written as words in the source ("thirty-minute", "forty five", "a hundred") count as given
    out=set(); t=text.lower().replace('-',' ')
    for m in re.finditer(r'\b(?:(%s)(?:\s+(%s))?|(%s)|(hundred|thousand|dozen))\b'%('|'.join(_TENS),'|'.join(list(_ONES)[1:10]),'|'.join(_ONES)),t):
        if m.group(1): out.add(float(_TENS[m.group(1)]+(_ONES[m.group(2)] if m.group(2) else 0)))
        elif m.group(3): out.add(float(_ONES[m.group(3)]))
        else: out.add({'hundred':100.0,'thousand':1000.0,'dozen':12.0}[m.group(4)])
    for i,w in enumerate('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth'.split(),1):
        if re.search(r'\b'+w+r'\b',t): out.add(float(i))
    return out
def numbers(text):
    out=[]
    for m in NUM.finditer(text):
        try: out.append(float(m.group().replace(',','')))
        except: pass
    return out
def first_json(text):
    t=re.sub(r'```(?:json)?','',text)
    starts=[k for k,ch in enumerate(t) if ch in '{[']
    for i in starts:
        op=t[i]; cl='}' if op=='{' else ']'; depth=0
        for j in range(i,len(t)):
            if t[j]==op: depth+=1
            elif t[j]==cl:
                depth-=1
                if depth==0:
                    try:
                        o=json.loads(t[i:j+1])
                        if isinstance(o,(dict,list)) and o: return o
                    except: pass
                    break
    return None
def flat(o):
    if isinstance(o,dict): return [x for v in o.values() for x in flat(v)]
    if isinstance(o,list): return [x for v in o for x in flat(v)]
    return [str(o).lower()]
ADMITS=["i would need","i'd need","i need more","i need to know","need more details","need more information","if you can provide","if you can share",
 "please describe","please share","please provide","please tell me","could you share","could you tell me","without seeing","can't","cannot","can not","unable","not able","no way to","no way of","no access","don't have","do not have","doesn't have",
 "don't know","do not know","not sure which","i'm not sure","not possible","isn't possible","can't see","don't see","do not see",
 "not attached","isn't attached","wasn't attached","no attachment","no photo","no image","no live","no real-time","not aware",
 "came through","didn't come through","didn't receive","haven't received","wasn't received","nothing came through","nothing was attached",
 "as an ai","i'm an ai","i am an ai","ai assistant","language model","text-based","beyond my","outside my","not something i can"]
EXPLAIN=re.compile(r'^\s*(?:#+\s*|\*\*)?(?:key\s+)?(?:fix(?:es)?|changes?(?: made)?|corrections?|explanation|what (?:i|was) changed|notes?|edits?)\b[^\n]*:?\s*(?:\*\*)?\s*$', re.I|re.M)
def _main(a):
    # the answer without its explanation of the changes: drop everything from a "Key fixes:" style heading,
    # and any line that shows a change ("sended" → "sent")
    m=EXPLAIN.search(a)
    if m and m.start()>0: a=a[:m.start()]
    return '\n'.join(l for l in a.splitlines() if not re.search(r'→|->|=>|\bchanged\b.*\bto\b|\breplaced\b.*\bwith\b', l, re.I))
STOP=set('a an the to of in on at for and or is are be my your his her its it this that with i you we me us our their them by from as'.split())
def _loose(text, v, window=4):
    # a fact said in another word form: every content word of it appears, as a word starting with its stem,
    # within a few words of each other ("bring back" ~ "bringing it back", "call again" ~ "call you again")
    words=[w for w in re.findall(r"[a-z0-9']+", v.lower()) if w not in STOP]
    if not words or (len(words)==1 and len(words[0])<5): return False
    toks=re.findall(r"[a-z0-9']+", _norm(text).lower())
    stems=[w[:max(4,len(w)-4)] if not w.isdigit() else w for w in words]
    pos=[[i for i,t in enumerate(toks) if (t==st if st.isdigit() else t.startswith(st))] for st in stems]
    if any(not p for p in pos): return False
    for start in pos[0]:
        if all(any(abs(i-start)<=window for i in p) for p in pos[1:]): return True
    return False
def run(check, answer, prompt):
    a=_norm(answer); t=check['type']; cs=check.get('case_sensitive',False)
    if t=='contains_any':
        ok=any(_has(a,v,cs) for v in check['values']); return ok, '' if ok else 'never mentions: '+', '.join(check['values'][:3])
    if t=='contains_all':
        miss=[v for v in check['values'] if not _has(a,v,cs)]; return not miss, 'missing: '+', '.join(miss) if miss else ''
    if t=='not_contains':
        a=_main(a)
        bad=[v for v in check['values'] if _has(a,v,cs)]; return not bad, 'still says: '+', '.join(bad) if bad else ''
    if t=='number':
        ok=any(abs(n-check['value'])<=check['tolerance'] for n in numbers(a)); return ok, '' if ok else f"didn't say {check['value']:g}"
    if t=='json':
        o=first_json(a)
        if o is None: return False,'not valid JSON'
        if check.get('only'):
            rest=re.sub(r'```(?:json)?','',a); i=min([k for k in (rest.find('{'),rest.find('[')) if k>=0]); depth=0
            op=rest[i]; cl='}' if op=='{' else ']'
            for j in range(i,len(rest)):
                depth+= (rest[j]==op)-(rest[j]==cl)
                if depth==0: break
            outside=(rest[:i]+' '+rest[j+1:]).strip()
            if outside and not re.fullmatch(r"[^\n]{0,60}:", outside) and not SIGNOFF.search(outside): return False,'wrote more than the JSON'
        vals=flat(o); miss=[alts[0] for alts in check['required_values'] if not any(x.lower() in v for x in alts for v in vals)]
        return not miss, 'missing: '+', '.join(miss) if miss else ''
    if t=='line_count':
        n=len(_lines(a)); return n==check['n'], f'{n} lines, expected {check["n"]}'
    if t=='max_words':
        n=len(a.split()); return n<=check['n'], f'{n} words, limit {check["n"]}'
    if t=='in_order':
        pos=0; low=_ampm(a).lower()
        for v in check['values']:
            alts=v if isinstance(v,list) else [v]
            hits=[m.start() for x in alts for m in re.finditer(r'(?<!\w)'+re.escape(_ampm(x).lower())+r'(?!\w)',low[pos:])]
            if not hits: return False, 'wrong order'
            pos+=min(hits)+1
        return True,''
    if t=='asks_back':
        ok='?' in a; return ok, '' if ok else "didn't ask what you meant"
    if t=='no_invented':
        pats={'phone':r'(?:\+?\d[\d\s-]{6,}\d)','url':r'https?://\S+|www\.\S+|\b[\w-]+\.(?:com|ae|net|org|io|co)/\S+','email':r'\b[\w.]+@[\w.]+\b',
              'price':r'(?:(?<!\w)(?:aed|dhs?|usd|eur|gbp)\s?\d)|(?:[$€£]\s?\d)|(?:\d\s?(?:k|m|mn|million|thousand)?\s?(?:aed|dhs|dirhams?|usd|eur|gbp|dollars?|euros?|pounds?)\b)',
              'distance':r'\d+(?:\.\d+)?\s?(?:km|kms|kilomet\w*|metres?|meters?|m\b|mins?\b|minutes?)'}
        dig=lambda x: re.sub(r'\D','',x)
        src=dig(prompt)
        for m in re.finditer(pats[check['what']],a,re.I):
            d=dig(m.group())
            if check['what'] in ('phone','price','distance') and d and d in src: continue
            if check['what'] in ('url','email') and m.group().lower() in prompt.lower(): continue
            return False, f"made up a {check['what']}"
        return True, ''
    if t=='numbers_from_source':
        # list markers ("1. ", "2) ") aren't claims, so they don't count; nor is a note of the answer's own length
        body=re.sub(r'(?m)^\s*(?:[-*•]\s*)?\d{1,2}[.)]\s+','',a)
        body=re.sub(r'\(?\s*(?:word count|words?)\s*[:=]?\s*\d+\s*\)?|\(\s*\d+\s*words?\s*\)|\b\d+\s*words?\b(?=\s*\)?\s*$)','',body,flags=re.I)
        # times are compared as times: "3 pm", "15:00" and "3:00 pm" are the same; "noon" is 12:00
        st=clock_times(_norm(prompt)); bad=[]
        def keep(m):
            h,mi,ap=_hm(m)
            ok=(h,mi) in st or (not ap and h<=12 and ((h+12)%24,mi) in st)
            if not ok: bad.append(m.group(0).strip())
            return ' '
        body=re.sub(r'\b(noon|midday)\b','12:00 pm',body,flags=re.I); body=re.sub(r'\bmidnight\b','12:00 am',body,flags=re.I)
        body=TIME.sub(keep,body)
        if bad: return False, 'invented the time '+bad[0]
        src=set(numbers(_norm(prompt)))|set(word_numbers(prompt))
        # "end of October" gives that month's last day
        for mo in re.findall(r'\bend of (january|february|march|april|may|june|july|august|september|october|november|december)\b', prompt, re.I):
            src.add({'february':28.0}.get(mo.lower(), 30.0 if mo.lower() in ('april','june','september','november') else 31.0))
        extra=[n for n in numbers(body) if n not in src]; return not extra, f'invented {extra[0]:g}' if extra else ''
    if t=='word_count':
        n=len(_body(a).split()); return n==check['n'], f'{n} words, expected {check["n"]}'
    if t=='sentence_count':
        b=re.sub(r'\b(?:dr|mr|mrs|ms|st|e\.g|i\.e|etc)\.', lambda m:m.group().replace('.',''), _body(a), flags=re.I)
        n=len([x for x in re.split(r'(?<=[.!?])\s+', b.strip()) if re.search(r'\w',x)]); return n==check['n'], f'{n} sentences, expected {check["n"]}'
    if t=='no_emoji':
        bad=re.search('[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B55\u2705\u274C\u2764\uFE0F]', a); return not bad, 'used an emoji' if bad else ''
    if t=='no_digits':
        bad=re.search(r'\d', a); return not bad, 'used a number' if bad else ''
    if t=='facts':
        # at least n of the listed facts; each fact is a list of ways to say it
        got=[f for f in check['values'] if any(_has(a,v,cs) or _loose(a,v) for v in f)]
        miss=[f[0] for f in check['values'] if f not in got]
        return len(got)>=check['n'], f"kept {len(got)} of {len(check['values'])} key facts, needs {check['n']} (missing: {', '.join(miss[:3])})"
    if t=='first_mention':
        # the right choice is named before any wrong one ("Nadia, not Nabil" passes; "Nabil" fails)
        low=_ampm(a).lower()
        def first(vals):
            ps=[m.start() for v in vals for m in re.finditer(r'(?<!\w)'+re.escape(_ampm(v).lower())+r'(?!\w)',low)]
            return min(ps) if ps else None
        r,w=first(check['right']),first(check['wrong'])
        if r is None: return False, 'never names '+check['right'][0]
        return (w is None or r<w), '' if (w is None or r<w) else 'names '+check['wrong'][0]+' first'
    if t=='admits_limit':
        # says it can't know or can't do it, in any of the usual ways
        ok=any(_has(a,v) for v in ADMITS); return ok, '' if ok else "never says it can't know or do this"
    if t=='any':
        # passes when any one of its checks passes ("asks what you meant, or says it can't know")
        res=[run(c,answer,prompt) for c in check['checks']]
        if any(r[0] is True for r in res): return True,''
        if any(r[0] is None for r in res): return None,'judge'
        return False, ' and '.join(r[1] for r in res)
    if t=='judge': return None,'judge'
    raise ValueError(t)
def mark(item, answer):
    res=[run(c,answer,item['prompt']) for c in item['checks']]
    if any(r[0] is False for r in res): return False, '; '.join(r[1] for r in res if r[0] is False)
    if any(r[0] is None for r in res): return None,'needs judge'
    return True,''
