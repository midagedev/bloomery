#!/usr/bin/env python3
"""Census of the GGUF files users download, by header read only.

For each repo it lists the files (HF API), groups them into sets (split shards
together, mmproj left out), and streams each shard's header over HTTPS until the
tensor infos end; no tensor byte is read. Writes census/<repo>__<set>.json
(arch, every tensor's type keyed by its name with the block index taken out,
the layers per type) and prints one line per set.

  census.py OUT_DIR [--stubs DIR] REPO...

With --stubs, each shard's header bytes (up to the tensor data, padded to the
file's alignment) are written to DIR/<repo>/<file> and the file is extended to
its real size, sparse: a stub the GGUF reader opens with every tensor in place
and every tensor byte zero, for plan-only probes.
"""
import collections, concurrent.futures as cf, json, os, re, struct, sys, urllib.request

T = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K',
     11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
     19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32',
     27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'}
SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
UA = {'User-Agent': 'bloomery-census/1'}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120)


def header(url):
    f = get(url)
    raw = []
    def rd(n):
        b = f.read(n)
        if len(b) != n:
            raise EOFError(url)
        raw.append(b)
        return b
    def u(fmt):
        return struct.unpack('<' + fmt, rd(struct.calcsize('<' + fmt)))[0]
    def s():
        return rd(u('Q')).decode('utf-8', 'replace')
    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u('I'), u('Q')
            if et == 8:
                for _ in range(n):
                    s()
            else:
                rd(n * struct.calcsize('<' + SZ[et]))
            return None
        return u(SZ[t])
    if rd(4) != b'GGUF':
        raise ValueError(f'{url}: not GGUF')
    u('I')
    nt, nk = u('Q'), u('Q')
    kv = {}
    for _ in range(nk):
        k = s(); t = u('I'); v = val(t)
        if k.startswith('general.') or k.endswith(('expert_count', 'expert_used_count', 'block_count',
                                                   'nextn_predict_layers', 'leading_dense_block_count')):
            if v is not None and not isinstance(v, str) or (isinstance(v, str) and len(v) < 200):
                kv[k] = v
    tensors = []
    for _ in range(nt):
        name = s(); nd = u('I'); dims = [u('Q') for _ in range(nd)]; ty = T.get(u('I'), '?'); u('Q')
        tensors.append((name, ty, dims))
    f.close()
    head = b''.join(raw)
    align = kv.get('general.alignment', 32)
    head += bytes(-len(head) % align)
    return kv, tensors, head


def sets_of(repo):
    resp = get(f'https://huggingface.co/api/models/{repo}/tree/main?recursive=1')
    if 'rel="next"' in (resp.headers.get('Link') or ''):
        raise ValueError(f'{repo}: the listing has a next page; this tool reads one')
    files = [(e['path'], (e.get('lfs') or {}).get('size', e.get('size'))) for e in json.load(resp)
             if e['type'] == 'file' and e['path'].endswith('.gguf') and 'mmproj' not in e['path'].lower()]
    sets = collections.defaultdict(list)
    for p, size in files:
        m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', p)
        sets[m.group(1) if m else p[:-5]].append((p, size))
    return {k: sorted(v) for k, v in sets.items()}


def census(repo, name, shards, out, stubs):
    kv, tensors, paths, sizes = {}, [], [p for p, _ in shards], {}
    for p, size in shards:
        k, t, head = header(f'https://huggingface.co/{repo}/resolve/main/{p}')
        kv.update(k); tensors += t; sizes[p] = size
        if stubs:
            fn = os.path.join(stubs, repo, p)
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, 'wb') as w:
                w.write(head)
                w.truncate(size)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    top = {}
    for n, ty, dims in tensors:
        m = re.match(r'blk\.(\d+)\.(.*)', n)
        if m:
            by[m.group(2)][ty].append(int(m.group(1)))
        else:
            top[n] = [ty, dims]
    rec = {'repo': repo, 'set': name, 'shards': paths, 'sizes': sizes, 'kv': kv, 'top': top,
           'blk': {k: {ty: sorted(ls) for ty, ls in v.items()} for k, v in sorted(by.items())}}
    fn = os.path.join(out, f"{repo.replace('/', '__')}__{os.path.basename(name)}.json")
    json.dump(rec, open(fn, 'w'), indent=1)
    return f"{repo} {os.path.basename(name)} arch={kv.get('general.architecture')} tensors={len(tensors)} shards={len(paths)}"


def main():
    out, repos, stubs = sys.argv[1], sys.argv[2:], None
    if repos[:1] == ['--stubs']:
        stubs, repos = repos[1], repos[2:]
    jobs = []
    for r in repos:
        try:
            for name, paths in sets_of(r).items():
                jobs.append((r, name, paths))
        except Exception as e:
            print(f'{r}: listing failed: {e}', flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(census, r, n, p, out, stubs): (r, n) for r, n, p in jobs}
        for fu in cf.as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:
                print(f'{futs[fu][0]} {futs[fu][1]}: FAILED {e}', flush=True)


if __name__ == '__main__':
    main()
