#!/usr/bin/env python3
"""Read tier-cost.sh's paired records. --self-test uses synthetic rows and a host-only launch probe."""
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloomery import records
from tdist import t975


def sample(source, schema):
    rows = records.of_kind(records.read(source, schema=schema), 'tier_cost')
    if len(rows) != 1 or rows[0].loose:
        raise ValueError('expected one complete tier_cost record')
    row = rows[0].fields
    expected = {'local': 1, 'node': 2}
    if row['arm'] not in expected or row['gpu_processes'] != expected[row['arm']]:
        raise ValueError('expected one local GPU process or two loopback GPU processes')
    if row['prompt'] < 1 or row['decode'] < 1 or len(row['tokens']) != row['prompt'] + row['decode']:
        raise ValueError('token count disagrees with prompt + decode')
    for name in ('tokens', 'history'):
        if not row[name] or any(type(t) is not int or t < 0 for t in row[name]):
            raise ValueError(f'{name} must contain token ids')
    for name in ('tg_tps', 'step_ms'):
        if not math.isfinite(row[name]) or row[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    return row


def summarize(pairs):
    if not pairs:
        raise ValueError('no local/node pairs')
    print('one machine, loopback: not a two-machine timing')
    reference = pairs[0][0]
    for r, (local, node) in enumerate(pairs, 1):
        if local['arm'] != 'local' or node['arm'] != 'node':
            raise ValueError(f'round {r}: arm order is not local/node')
        for name in ('plan', 'history', 'tokens', 'prompt', 'decode'):
            if local[name] != node[name] or local[name] != reference[name]:
                raise ValueError(f'round {r}: {name} differs between arms or rounds')
        print(f'PAIR r{r} same_plan=true same_history=true same_tokens=true '
              f'local_gpu_processes={local["gpu_processes"]} '
              f'node_gpu_processes={node["gpu_processes"]}')
    if len(pairs) < 2:
        return
    for metric in ('tg_tps', 'step_ms'):
        ratios = [b[metric] / a[metric] for a, b in pairs]
        mean = statistics.mean(ratios)
        half = t975(len(ratios) - 1) * statistics.stdev(ratios) / math.sqrt(len(ratios))
        print(f'ratio {metric} node/local={mean:.6f} ci95=[{mean-half:.6f},{mean+half:.6f}] '
              f'relative_pct={100*(mean-1):.6f} pairs={len(ratios)} paired_t_df={len(ratios)-1} [derived]')


def measurement_host_room_test():
    from recipes import _close, rust_lex

    bins = Path(__file__).resolve().parents[2] / 'crates/gpu-gates/src/bin'
    source = (bins / 'gate_qwen38_nodes.rs').read_text()
    shared = (bins / 'shared/node_proc.rs').read_text()

    def function(name, text=source):
        _, shape = rust_lex(text)
        start = re.search(r'\bfn ' + name + r'\(', shape).start()
        end = _close(shape, shape.index('{', start))
        assert end is not None, f'{name}: unclosed body'
        return text[start:end + 1]

    run = function('run')
    branch = run.index('} else {', run.index('if arm == "local" {')) + len('} else {')
    launch = run[branch:run.index('let mut child = node0(', branch)]
    room = re.search(r'const GATE_HOST_ROOM: u64 = [^;]+;', source).group()
    # Execute the measurement caller and command builder; stop at the process boundary.
    probe = r'''
use std::{net::SocketAddr, path::{Path, PathBuf}, process::Command, time::Duration};
type GateError = Box<dyn std::error::Error>;
const TIER_NODES: &str = "BLOOMERY_TIER_NODES";
const CARD_BUDGET: &str = "BLOOMERY_CARD_BUDGET";
const HOST_POPULATE: &str = "BLOOMERY_HOST_POPULATE";
const HOST_LOCK: &str = "BLOOMERY_HOST_LOCK";
const CARD_DONTNEED: &str = "BLOOMERY_CARD_DONTNEED";
mod bloomery_levers {
    pub const HOST_ROOM: &str = "BLOOMERY_HOST_ROOM";
    pub struct NvTierLevers { pub host_room: Option<u64> }
    pub fn nvtier_levers() -> Result<NvTierLevers, super::GateError> {
        Ok(NvTierLevers {
            host_room: std::env::var_os(HOST_ROOM)
                .map(|v| v.to_str().unwrap().parse()).transpose()?,
        })
    }
}
mod bloomery_gpu_gates {
    pub fn beside(name: &str) -> Result<std::path::PathBuf, super::GateError> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }
}
struct Common { node_host: bool }
struct Proc;
impl Proc {
    fn spawn(cmd: Command, _: &str) -> Result<Self, GateError> {
        let inherited = std::env::var_os(bloomery_levers::HOST_ROOM);
        let actual = cmd.get_envs()
            .find(|(key, _)| *key == bloomery_levers::HOST_ROOM)
            .map_or(inherited.as_deref(), |(_, value)| value);
        assert_eq!(actual, inherited.as_deref(), "measurement host room");
        Err("launch inspected".into())
    }
    fn wait_line(&self, _: &str, _: Duration) -> Result<String, GateError> { unreachable!() }
}
'''
    consts = '\n'.join(re.search(r'pub const ' + name + r': [^;]+;', shared).group()
                        for name in ('LISTENING', 'BIND_WITHIN'))
    probe += room + '\n' + consts + '\n' + function('tier_server_command', shared)
    probe += '\n' + function('tier_server', shared) + '\n' + function('node1_command')
    probe += '\n' + function('node1')
    probe += '\nfn measurement(node_host: bool) -> Result<(), GateError> {\n'
    probe += 'let c = Common { node_host };\n' + launch + '\nOk(())\n}\n'
    probe += ('fn main() { for host in [false, true] {\n'
              'assert_eq!(measurement(host).unwrap_err().to_string(), "launch inspected");\n'
              '} }\n')
    with tempfile.TemporaryDirectory(prefix='tier-cost-room-') as name:
        path = Path(name)
        (path / 'probe.rs').write_text(probe)
        build = subprocess.run(['rustc', '--edition=2024', str(path / 'probe.rs'),
                                '-o', str(path / 'probe')], capture_output=True,
                               text=True, timeout=120)
        assert build.returncode == 0, build.stderr
        for value in (str(32 << 30), '0', None):
            env = dict(os.environ)
            if value is None:
                env.pop('BLOOMERY_HOST_ROOM', None)
            else:
                env['BLOOMERY_HOST_ROOM'] = value
            result = subprocess.run([str(path / 'probe')], env=env,
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, f'host room {value}: {result.stderr}'
    print('tier-cost measurement host room: unset, zero and 32 GiB preserved (host-only)')


def self_test():
    measurement_host_room_test()
    schema = records.load('gate_qwen38_nodes', Path(__file__).with_name('data') / 'tier-cost.schema.jsonl')
    line = ('tier cost arm=local plan=ctx=3072 ub=8 stage=[1] tier=[[1]] host=1 '
            'history=[1, 2] tokens=[3, 4] prompt=1 decode=1 tg_tps=100.000000 '
            'step_ms=10.000000 gpu_processes=1 budget_gib=64 elapsed_s=0.010000000 '
            'control_round_trips=0 handoffs=2 read_terms=Some(Terms { hbm: 4096 }) '
            'wire=[(128, [64, 64])] hits_before=[[1, 0]] hits_after=[[2, 1]]')
    local = sample([line], schema)
    assert [local[k] for k in ('budget_gib', 'elapsed_s', 'control_round_trips', 'handoffs')] == [64, 0.01, 0, 2]
    assert [local[k] for k in ('read_terms', 'wire', 'hits_before', 'hits_after')] == [
        'Some(Terms { hbm: 4096 })', '[(128, [64, 64])]', '[[1, 0]]', '[[2, 1]]']
    node = dict(local, arm='node', tg_tps=80.0, step_ms=12.5, gpu_processes=2)
    pairs = [(local, node), (local, node)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        summarize(pairs)
    assert 'same_tokens=true' in out.getvalue()
    assert 'ratio tg_tps node/local=0.800000 ci95=[0.800000,0.800000]' in out.getvalue()
    assert 'ratio step_ms node/local=1.250000 ci95=[1.250000,1.250000]' in out.getvalue()
    changed = [dict(local, tokens=[3, 5]), dict(node, tokens=[3, 5])]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            summarize([(local, node), changed])
    except ValueError:
        pass
    else:
        raise AssertionError('cross-round tokens mismatch accepted')
    varied = [(local, node), (local, dict(node, tg_tps=100.0, step_ms=10.0))]
    with contextlib.redirect_stdout(out := io.StringIO()):
        summarize(varied)
    assert 'ratio tg_tps node/local=0.900000 ci95=[-0.370600,2.170600]' in out.getvalue()
    cases = 3
    for field, value in (('tokens', [3, 5]), ('history', [1, 3]), ('plan', 'another'),
                         ('prompt', 2), ('decode', 2), ('arm', 'local')):
        bad = copy.deepcopy(pairs)
        bad[1][1][field] = value
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(bad)
        except ValueError:
            cases += 1
        else:
            raise AssertionError(f'{field} mismatch accepted')
    for old, new in (('tokens=[3, 4]', 'tokens=[3]'), ('tg_tps=100.000000', 'tg_tps=NaN'),
                     ('step_ms=10.000000', 'step_ms=0.000000'), ('gpu_processes=1', 'gpu_processes=2')):
        try:
            sample([line.replace(old, new)], schema)
        except ValueError:
            cases += 1
        else:
            raise AssertionError(f'malformed sample accepted: {new}')
    for lines in ([], [line, line], [line.replace(' decode=1', '')],
                  [line.replace(' hits_after=[[2, 1]]', '')]):
        try:
            sample(lines, schema)
        except ValueError:
            cases += 1
        else:
            raise AssertionError('missing or duplicate record accepted')
    print(f'tier-cost self-test: {cases} of {cases} ok (synthetic, NOT TIMING)')


def main(argv):
    if argv == ['--self-test']:
        self_test()
        return
    if len(argv) != 2 or not argv[1].isdigit() or not 1 <= int(argv[1]) <= 100:
        raise ValueError('usage: tier-cost.py LOG_DIRECTORY ROUNDS | --self-test')
    root = Path(argv[0])
    schema = records.load('gate_qwen38_nodes', root / 'schema.jsonl')
    pairs = []
    for r in range(1, int(argv[1]) + 1):
        pair = []
        for arm in ('local', 'node'):
            pair.append(sample(root / f'r{r}-{arm}.log', schema))
        pairs.append(pair)
    summarize(pairs)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except (ValueError, OSError) as error:
        print(f'tier-cost: {error}', file=sys.stderr)
        sys.exit(1)
