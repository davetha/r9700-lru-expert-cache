import json, urllib.request, random, sys

BASE = 'http://127.0.0.1:8057'
MODEL = 'q38fn-mxfp4'

def post(path, payload, timeout=2400):
    return urllib.request.urlopen(urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}), timeout=timeout)

SUBJ = ['The inlet sensor', 'A coolant pump', 'The reserve fan', 'Bay 4 telemetry',
        'The intake filter', 'A pressure valve', 'The chiller loop', 'Rack 12 airflow']
VERB = ['reported', 'logged', 'flagged', 'recorded', 'showed', 'indicated']

def filler(n_lines, seed):
    r = random.Random(seed)
    return [f'{r.choice(SUBJ)} {r.choice(VERB)} a delta of {r.randrange(1,99)}.{r.randrange(10,99)} degrees.'
            for _ in range(n_lines)]

# distinctive, unguessable needle
NEEDLE = ('IMPORTANT MAINTENANCE RECORD: technician Marguerite Oyelaran replaced '
          'the auxiliary manifold gasket, part number QX-7731-B, on service ticket 48210.')
ANSWERS = {'part': 'QX-7731-B', 'ticket': '48210', 'name': 'Oyelaran'}

def build(n_lines, depth, seed):
    lines = filler(n_lines, seed)
    pos = int(len(lines) * depth)
    lines.insert(pos, NEEDLE)
    return 'Below is a maintenance log.\n\n' + '\n'.join(lines) + \
           '\n\nQuestion: what is the part number and the service ticket number in the IMPORTANT MAINTENANCE RECORD, and who was the technician? Answer with just those three values.'

def ask(prompt):
    d = json.load(post('/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 120, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False}}))
    m = d['choices'][0]['message']
    return (m.get('content') or '').strip(), d['usage']['prompt_tokens']

def run(n_lines, label, seed):
    for depth, dl in [(0.1, '10%'), (0.5, '50%'), (0.9, '90%')]:
        p = build(n_lines, depth, seed)
        try:
            txt, pt = ask(p)
        except Exception as e:
            print(f'  {label:6} depth={dl:4} FAILED {type(e).__name__} {str(e)[:70]}')
            continue
        hits = [k for k, v in ANSWERS.items() if v.lower() in txt.lower()]
        verdict = 'PASS' if len(hits) == 3 else ('PARTIAL' if hits else 'FAIL')
        print(f'  {verdict:7} {label:6} depth={dl:4} prompt={pt:>7} found={sorted(hits)} -> {txt[:70]!r}')

for n, lab, sd in [(2200, '32K', 11), (9000, '128K', 12), (15500, '200K', 13)]:
    run(n, lab, sd)
