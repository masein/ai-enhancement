import re, json
NUM = re.compile(r'(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?')
def _norm(s): return s.replace('’',"'").replace('½',' 1/2')
def _ampm(x): return re.sub(r'(\d)\s*(am|pm|a\.m\.|p\.m\.)(?!\w)', r'\1 \2', x, flags=re.I)
def _has(text, v, cs=False):
    text, v = _ampm(text), _ampm(v)
    t, v = (text, v) if cs else (text.lower(), v.lower())
    return re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)', t) is not None
def _body(a):
    ls=[l for l in a.splitlines() if l.strip() and not l.strip().startswith('```')]
    if len(ls)>1 and ls[0].rstrip().endswith(':'): ls=ls[1:]
    return '\n'.join(ls)
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
def run(check, answer, prompt):
    a=_norm(answer); t=check['type']; cs=check.get('case_sensitive',False)
    if t=='contains_any':
        ok=any(_has(a,v,cs) for v in check['values']); return ok, '' if ok else 'says none of: '+', '.join(check['values'][:3])
    if t=='contains_all':
        miss=[v for v in check['values'] if not _has(a,v,cs)]; return not miss, 'missing: '+', '.join(miss) if miss else ''
    if t=='not_contains':
        bad=[v for v in check['values'] if _has(a,v,cs)]; return not bad, 'still says: '+', '.join(bad) if bad else ''
    if t=='number':
        ok=any(abs(n-check['value'])<=check['tolerance'] for n in numbers(a)); return ok, '' if ok else f"didn't say {check['value']:g}"
    if t=='json':
        o=first_json(a)
        if o is None: return False,'not valid JSON'
        vals=flat(o); miss=[alts[0] for alts in check['required_values'] if not any(x.lower() in v for x in alts for v in vals)]
        return not miss, 'missing: '+', '.join(miss) if miss else ''
    if t=='line_count':
        ls=[l for l in a.splitlines() if l.strip() and not l.strip().startswith('```')]
        if len(ls)>1 and ls[0].rstrip().endswith(':'): ls=ls[1:]
        n=len(ls); return n==check['n'], f'{n} lines, expected {check["n"]}'
    if t=='max_words':
        n=len(a.split()); return n<=check['n'], f'{n} words, limit {check["n"]}'
    if t=='in_order':
        pos=0; low=a.lower()
        for v in check['values']:
            alts=v if isinstance(v,list) else [v]
            hits=[m.start() for x in alts for m in re.finditer(r'(?<!\w)'+re.escape(x.lower())+r'(?!\w)',low[pos:])]
            if not hits: return False, 'wrong order'
            pos+=min(hits)+1
        return True,''
    if t=='asks_back':
        ok='?' in a; return ok, '' if ok else "didn't ask what you meant"
    if t=='no_invented':
        pats={'phone':r'(?:\+?\d[\d\s-]{6,}\d)','url':r'https?://|www\.|\b\w+\.(?:com|ae|net|org)\b','email':r'\b[\w.]+@[\w.]+\b',
              'price':r'(?:\b(?:aed|dhs?|usd|\$)\s?\d)|(?:\d\s?(?:k|m|mn|million|thousand)?\s?(?:aed|dhs|dirhams?|usd)\b)',
              'distance':r'\d+(?:\.\d+)?\s?(?:km|kms|kilomet\w*|metres?|meters?|m\b|mins?\b|minutes?)'}
        bad=re.search(pats[check['what']],a,re.I); return not bad, f"made up a {check['what']}" if bad else ''
    if t=='numbers_from_source':
        src=set(numbers(_norm(prompt))); extra=[n for n in numbers(a) if n not in src]; return not extra, f'invented {extra[0]:g}' if extra else ''
    if t=='word_count':
        n=len(_body(a).split()); return n==check['n'], f'{n} words, expected {check["n"]}'
    if t=='sentence_count':
        b=re.sub(r'\b(?:dr|mr|mrs|ms|st|e\.g|i\.e|etc)\.', lambda m:m.group().replace('.',''), _body(a), flags=re.I)
        n=len([x for x in re.split(r'(?<=[.!?])\s+', b.strip()) if re.search(r'\w',x)]); return n==check['n'], f'{n} sentences, expected {check["n"]}'
    if t=='no_emoji':
        bad=re.search('[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B55\u2705\u274C\u2764\uFE0F]', a); return not bad, 'used an emoji' if bad else ''
    if t=='no_digits':
        bad=re.search(r'\d', a); return not bad, 'used a number' if bad else ''
    if t=='judge': return None,'judge'
    raise ValueError(t)
def mark(item, answer):
    res=[run(c,answer,item['prompt']) for c in item['checks']]
    if any(r[0] is False for r in res): return False, '; '.join(r[1] for r in res if r[0] is False)
    if any(r[0] is None for r in res): return None,'needs judge'
    return True,''
