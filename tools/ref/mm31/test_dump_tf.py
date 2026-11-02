"""Tests of tools/ref/mm31/dump_tf.py on a synthetic MiniMax-M3.1-like folder: the vendor's tensor names, dtypes and layout
at a size a laptop runs in seconds (3 layers: a dense FFN and two sparse ones, 8 experts, 4 routed per token, an MSA
indexer of 2 heads over blocks of 4 with a top-3, so a prompt past 12 positions drops blocks).

    python tools/ref/mm31/test_dump_tf.py            (the venv of requirements.txt)

The reference is transformers' own MiniMaxM3VLForCausalLM run on weights assembled here through compressed-tensors' and
Fp8Dequantize's dequantization, a path that shares no code with the dumper's layer-streamed one. Each mutant is a named
change of the dumper that the matching test must catch: the test passes only when the mutant is refused or the
comparison goes red."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dump_tf", os.path.join(HERE, "dump_tf.py"))
dump_tf = importlib.util.module_from_spec(spec)
sys.modules["dump_tf"] = dump_tf
spec.loader.exec_module(dump_tf)

HID, HEADS, KV, HD = 64, 8, 2, 16
INTER, DENSE, SHARED = 32, 64, 32
EXPERTS, TOPK, VOCAB, LAYERS = 8, 4, 128, 3
IDX_HEADS, IDX_DIM, BLOCK, TOPK_BLOCKS = 2, 16, 4, 3
PROMPTS = {"a": 10, "b": 14, "c": 20, "d": 24, "e": 26}
PFX = "language_model.model."

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_mxfp8(w):
    """e4m3 values and one E8M0 byte per 32 values of a row."""
    n, k = w.shape
    blocks = w.reshape(n, k // 32, 32)
    amax = blocks.abs().amax(-1)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.zeros_like(amax)).clamp(-126, 127)
    q = (blocks / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).reshape(n, k)
    return q, (e + 127).to(torch.uint8)


def quantize_nvfp4(w):
    """packed bytes (value 2i low, 2i + 1 high), E4M3 scales per 16 values, the global divisor."""
    n, k = w.shape
    g = torch.tensor(448.0 * 6.0 / float(w.abs().max()))
    blocks = w.reshape(n, k // 16, 16)
    sc = (blocks.abs().amax(-1) / 6.0 * g).clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = sc.float() / g
    q = torch.where(eff[..., None] > 0, blocks / eff[..., None].clamp(min=1e-30), torch.zeros_like(blocks))
    idx = (q.abs()[..., None] - MAGS).abs().argmin(-1)
    nib = (idx | ((q < 0).to(torch.int64) << 3)).reshape(n, k).to(torch.uint8)
    return nib[:, 0::2] | (nib[:, 1::2] << 4), sc, g


# What a cut's first shard carries: the cutter's identity keys, the manifest's hash and the card budget the line leaves out.
CUT_KEYS = {
    "bloomery.fixture.version": "1",
    "bloomery.fixture.source": "synthetic",
    "bloomery.fixture.rev": dump_tf.REV,
    "bloomery.fixture.layers": "0,1,2",
    "bloomery.fixture.ff": str(INTER),
    "bloomery.fixture.manifest_sha256": "ab" * 32,
    "bloomery.fixture.card_budget": "30000000000",
}


def cut_line(folder):
    """refset's `# fixture` line of a cut, as `refset-check --fixture-line` prints it: the first shard's bloomery.fixture.* keys
    sorted, the card budget left out; None for a folder that is not a cut."""
    from safetensors import safe_open

    with safe_open(os.path.join(folder, "model-00001-of-00002.safetensors"), "pt") as f:
        meta = f.metadata() or {}
    keys = sorted(k for k in meta if k.startswith("bloomery.fixture.") and k != "bloomery.fixture.card_budget")
    return "# fixture\t" + " ".join("%s=%s" % (k, meta[k]) for k in keys) if keys else None


def make_folder(path, seed=0, fixture=True):
    """Write the synthetic folder under `path`; returns the tensors it holds by checkpoint name."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s, std=0.1: torch.randn(*s, generator=g) * std
    t = {}

    def mx(name, n, k, std=0.1):
        q, s = quantize_mxfp8(rnd(n, k, std=std))
        t[name + ".weight"], t[name + ".weight_scale_inv"] = q, s

    t[PFX + "embed_tokens.weight"] = rnd(VOCAB, HID, std=1.0).to(torch.bfloat16)
    t[PFX + "norm.weight"] = rnd(HID, std=0.1).to(torch.bfloat16)
    t["language_model.lm_head.weight"] = rnd(VOCAB, HID, std=0.3).to(torch.bfloat16)
    for L in range(LAYERS):
        P = "%slayers.%d." % (PFX, L)
        for n in ("input_layernorm", "post_attention_layernorm"):
            t[P + n + ".weight"] = rnd(HID, std=0.1).to(torch.bfloat16)
        mx(P + "self_attn.q_proj", HEADS * HD, HID)
        mx(P + "self_attn.k_proj", KV * HD, HID)
        mx(P + "self_attn.v_proj", KV * HD, HID)
        mx(P + "self_attn.o_proj", HID, HEADS * HD)
        mx(P + "self_attn.index_q_proj", IDX_HEADS * IDX_DIM, HID, std=0.4)
        mx(P + "self_attn.index_k_proj", IDX_DIM, HID, std=0.4)
        for n, d in (("q_norm", HD), ("k_norm", HD), ("index_q_norm", IDX_DIM), ("index_k_norm", IDX_DIM)):
            t[P + "self_attn." + n + ".weight"] = rnd(d, std=0.1).to(torch.bfloat16)
        if L == 0:
            mx(P + "mlp.gate_proj", DENSE, HID)
            mx(P + "mlp.up_proj", DENSE, HID)
            mx(P + "mlp.down_proj", HID, DENSE)
            continue
        M = P + "block_sparse_moe."
        t[M + "gate.weight"] = rnd(EXPERTS, HID, std=0.5)
        t[M + "e_score_correction_bias"] = rnd(EXPERTS, std=0.05)
        mx(M + "shared_experts.gate_proj", SHARED, HID)
        mx(M + "shared_experts.up_proj", SHARED, HID)
        mx(M + "shared_experts.down_proj", HID, SHARED)
        for e in range(EXPERTS):
            for w, n, k in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
                packed, sc, gs = quantize_nvfp4(rnd(n, k, std=0.1))
                S = "%sexperts.%d.%s." % (M, e, w)
                t[S + "weight_packed"], t[S + "weight_scale"], t[S + "weight_global_scale"] = packed, sc, gs
                t[S + "input_global_scale"] = torch.tensor(1.0)
    os.makedirs(path, exist_ok=True)
    shard_of = lambda name: "model-00001-of-00002.safetensors" if ".layers.1." not in name and ".layers.2." not in name else "model-00002-of-00002.safetensors"
    weight_map = {k: shard_of(k) for k in t}
    for shard in sorted(set(weight_map.values())):
        meta = {"format": "pt"}
        if fixture and shard.startswith("model-00001"):
            meta.update(CUT_KEYS)
        save_file({k: v.contiguous() for k, v in t.items() if weight_map[k] == shard}, os.path.join(path, shard), metadata=meta)
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(path, "model.safetensors.index.json"), "w"))
    cfg = {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": HID, "intermediate_size": INTER, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV, "head_dim": HD, "vocab_size": VOCAB, "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "rotary_dim": HD // 2, "partial_rotary_factor": 0.5,
            "hidden_act": "swigluoai", "dense_intermediate_size": DENSE, "shared_intermediate_size": SHARED,
            "num_local_experts": EXPERTS, "num_experts_per_tok": TOPK, "n_shared_experts": 1, "scoring_func": "sigmoid",
            "use_routing_bias": True, "moe_layer_freq": [0, 1, 1], "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "routed_scaling_factor": 2.0, "tie_word_embeddings": False, "bos_token_id": None, "eos_token_id": None,
            "sparse_attention_config": {
                "use_sparse_attention": True, "sparse_index_dim": IDX_DIM, "sparse_num_index_heads": IDX_HEADS,
                "sparse_topk_blocks": TOPK_BLOCKS, "sparse_block_size": BLOCK, "sparse_local_block": 1,
                "sparse_attention_freq": [1, 1, 1],
            },
        },
    }
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    return t


def make_ids(path, seed=1):
    os.makedirs(path, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for name, n in PROMPTS.items():
        ids = torch.randint(0, VOCAB, (n,), generator=g).tolist()
        open(os.path.join(path, name + ".ids"), "w").write("\n".join(map(str, ids)) + "\n")


def read_ids(path, name):
    return [int(x) for x in open(os.path.join(path, name + ".ids")).read().split()]


def f32(path, name, occ=0):
    return np.fromfile(os.path.join(path, "%s.%d.f32" % (name, occ)), dtype="<f4")


def dequantized_state(folder, cfg):
    """The vendor model's state dict on weights dequantized by compressed-tensors' and transformers' own functions."""
    sd = {}
    with safe_open(os.path.join(folder, "model-00001-of-00002.safetensors"), "pt") as f1, safe_open(
        os.path.join(folder, "model-00002-of-00002.safetensors"), "pt"
    ) as f2:
        def get(name):
            for f in (f1, f2):
                if name in f.keys():
                    return f.get_tensor(name)
            raise KeyError(name)

        def mx(stem):
            return dump_tf.mxfp8_reference(get(stem + ".weight").view(torch.uint8), get(stem + ".weight_scale_inv"))

        def nv(stem):
            return dump_tf.nvfp4_reference(
                get(stem + ".weight_packed"), get(stem + ".weight_scale").view(torch.uint8), get(stem + ".weight_global_scale")
            )[0]

        sd["model.embed_tokens.weight"] = get(PFX + "embed_tokens.weight").float()
        sd["model.norm.weight"] = get(PFX + "norm.weight").float()
        sd["lm_head.weight"] = get("language_model.lm_head.weight").float()
        for L in range(cfg.num_hidden_layers):
            P, Q = "%slayers.%d." % (PFX, L), "model.layers.%d." % L
            for src, dst in (("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm")):
                sd[Q + dst + ".weight"] = get(P + src + ".weight").float()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sd[Q + "self_attn.%s.weight" % n] = mx(P + "self_attn." + n)
            for n in ("q_norm", "k_norm"):
                sd[Q + "self_attn.%s.weight" % n] = get(P + "self_attn." + n + ".weight").float()
            for n in ("q", "k"):
                sd[Q + "self_attn.indexer.%s_proj.weight" % n] = mx(P + "self_attn.index_%s_proj" % n)
                sd[Q + "self_attn.indexer.%s_norm.weight" % n] = get(P + "self_attn.index_%s_norm.weight" % n).float()
            if cfg.mlp_layer_types[L] == "dense":
                sd[Q + "mlp.gate_up_proj.weight"] = torch.cat([mx(P + "mlp.gate_proj"), mx(P + "mlp.up_proj")], 0)
                sd[Q + "mlp.down_proj.weight"] = mx(P + "mlp.down_proj")
                continue
            M = P + "block_sparse_moe."
            sd[Q + "mlp.gate.weight"] = get(M + "gate.weight")
            sd[Q + "mlp.gate.e_score_correction_bias"] = get(M + "e_score_correction_bias")
            sd[Q + "mlp.shared_experts.gate_up_proj.weight"] = torch.cat(
                [mx(M + "shared_experts.gate_proj"), mx(M + "shared_experts.up_proj")], 0
            )
            sd[Q + "mlp.shared_experts.down_proj.weight"] = mx(M + "shared_experts.down_proj")
            sd[Q + "mlp.experts.gate_up_proj"] = torch.stack(
                [torch.cat([nv("%sexperts.%d.w1" % (M, e)), nv("%sexperts.%d.w3" % (M, e))], 0) for e in range(EXPERTS)]
            )
            sd[Q + "mlp.experts.down_proj"] = torch.stack([nv("%sexperts.%d.w2" % (M, e)) for e in range(EXPERTS)])
    return sd


class Reference:
    """transformers' MiniMaxM3VLForCausalLM on the dequantized weights, with the taps the tests compare."""

    def __init__(self, folder):
        self.hf = dump_tf.load_vendor()
        self.cfg, _ = dump_tf.text_config(folder)
        self.model = self.hf.MiniMaxM3VLForCausalLM(self.cfg).eval()
        self.model.load_state_dict(dequantized_state(folder, self.cfg), strict=True)

    def forward(self, ids):
        outs = []
        hooks = [layer.register_forward_hook(lambda m, a, o: outs.append(o.detach().clone())) for layer in self.model.model.layers]
        try:
            logits = self.model(torch.tensor([ids]), use_cache=False).logits[0]
        finally:
            for h in hooks:
                h.remove()
        return logits, outs

    def greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(self.model(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax()))
        return ids[-n:]


def dump(folder, ids, out, extra=(), line=True):
    """The dumper on `folder`; a cut gets refset's line unless `line` is False (the runner's duty, here the test's)."""
    args = ["--folder", folder, "--out", out, "--ids", ids, "--greedy", "6", *extra]
    if line and "--fixture-line" not in args and cut_line(folder) is not None:
        args += ["--fixture-line", cut_line(folder)]
    return dump_tf.main(args)


class DequantTests(unittest.TestCase):
    def test_both_rules_equal_their_references_bit_for_bit(self):
        g = torch.Generator().manual_seed(3)
        w = torch.randn(48, 96, generator=g) * 0.2
        packed, sc, gs = quantize_nvfp4(w)
        sc_u8 = sc.view(torch.uint8)
        ref, _ = dump_tf.nvfp4_reference(packed, sc_u8, gs)
        self.assertTrue(dump_tf.same_bits(dump_tf.nvfp4(packed, sc_u8, gs), ref))
        dump_tf.check_nvfp4("synthetic", packed, sc_u8, gs)
        q, s = quantize_mxfp8(w)
        dump_tf.check_mxfp8("synthetic", q, s)
        # the quantizers are sane: the dequantized matrix is near the source (e2m1 steps are coarse)
        self.assertLess(float((dump_tf.nvfp4(packed, sc_u8, gs) - w).abs().mean()), 0.06)
        self.assertLess(float((dump_tf.mxfp8(q.view(torch.uint8), s) - w).abs().mean()), 0.01)

    def test_a_dequantization_that_multiplies_the_global_scale_is_refused(self):
        """Mutant: scale x g where compressed-tensors divides by it."""
        g = torch.Generator().manual_seed(4)
        packed, sc, gs = quantize_nvfp4(torch.randn(16, 64, generator=g) * 0.2)
        good = dump_tf.nvfp4

        def mutant(p, s, gscale, out=None):
            return good(p, s, 1.0 / gscale, out=out)

        dump_tf.nvfp4 = mutant
        try:
            with self.assertRaisesRegex(dump_tf.Refusal, "differs from compressed-tensors"):
                dump_tf.check_nvfp4("mutant", packed, sc.view(torch.uint8), gs)
        finally:
            dump_tf.nvfp4 = good

    def test_an_mxfp8_that_reads_the_scale_as_a_plain_byte_is_refused(self):
        """Mutant: the E8M0 byte taken as a multiplier, not 2^(byte - 127)."""
        q, s = quantize_mxfp8(torch.randn(8, 64) * 0.2)
        good = dump_tf.mxfp8

        def mutant(w, sc, out=None):
            return good(w, sc + 1, out=out)

        dump_tf.mxfp8 = mutant
        try:
            with self.assertRaisesRegex(dump_tf.Refusal, "differs from transformers"):
                dump_tf.check_mxfp8("mutant", q, s)
        finally:
            dump_tf.mxfp8 = good

    def test_corrupt_bytes_are_refused(self):
        stats = dump_tf.Stats()
        with self.assertRaisesRegex(dump_tf.Refusal, "sign bit"):
            dump_tf.scale_check("s", torch.tensor([0x7F], dtype=torch.uint8), stats)
        with self.assertRaisesRegex(dump_tf.Refusal, "sign bit"):
            dump_tf.scale_check("s", torch.tensor([0x81], dtype=torch.uint8), stats)
        dump_tf.scale_check("s", torch.tensor([0, 1, 0x7E], dtype=torch.uint8), stats)
        self.assertEqual(stats["zero_scales"], 1)
        with self.assertRaisesRegex(dump_tf.Refusal, "NaN bytes"):
            dump_tf.mxfp8_check("w", torch.tensor([[0x7F] * 32], dtype=torch.uint8).view(torch.float8_e4m3fn), torch.zeros(1, 1, dtype=torch.uint8))
        with self.assertRaisesRegex(dump_tf.Refusal, "NaN bytes"):
            dump_tf.mxfp8_check("w", torch.zeros(1, 32, dtype=torch.uint8).view(torch.float8_e4m3fn), torch.full((1, 1), 0xFF, dtype=torch.uint8))


class Oracle(unittest.TestCase):
    """One synthetic folder, dumped once; the tests read the sets."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mm31-test-")
        cls.folder = os.path.join(cls.tmp, "model")
        cls.ids = os.path.join(cls.tmp, "ids")
        cls.out = os.path.join(cls.tmp, "out")
        make_folder(cls.folder)
        make_ids(cls.ids)
        cls.rc = dump(cls.folder, cls.ids, cls.out)
        cls.ref = Reference(cls.folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def set(self, name):
        return os.path.join(self.out, "ref_minimaxm31_" + name)

    def test_the_dump_ran_and_every_set_is_complete(self):
        self.assertEqual(self.rc, 0)
        for name in ("a", "b", "c", "d", "e", "e_perhead"):
            text = open(os.path.join(self.set(name), "MANIFEST.tsv")).read()
            self.assertIn("# complete\t", text, name)
            self.assertIn("# arch\tminimax-m3\n", text)
            self.assertTrue(os.path.isfile(os.path.join(self.set(name), "topk.tsv")))
            self.assertTrue(os.path.isfile(os.path.join(self.set(name), "greedy.tsv")))

    def test_the_build_line_names_the_writer_by_its_md5_and_the_public_library_versions(self):
        import hashlib
        import torch
        import transformers

        md5 = hashlib.md5(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_tf.py"), "rb").read()).hexdigest()
        want = "transformers-%s+torch-%s+dump_tf-%s" % (transformers.__version__, str(torch.__version__).split("+")[0], md5[:12])
        for name in ("a", "e_perhead"):
            lines = open(os.path.join(self.set(name), "MANIFEST.tsv")).read().splitlines()
            self.assertIn("# build\t" + want, lines, name)
            self.assertIn("# torch\t" + torch.__version__, lines, name)
            model = next(l for l in lines if l.startswith("# model\t")).split("\t")[1]
            self.assertEqual(os.path.dirname(model), self.folder, name)
            self.assertTrue(os.path.basename(model).startswith("model-00001-of-"), model)

    def test_a_cut_states_refsets_fixture_line_and_its_first_shard(self):
        want = cut_line(self.folder)
        self.assertNotIn("card_budget", want)
        for name in ("a", "e_perhead"):
            lines = open(os.path.join(self.set(name), "MANIFEST.tsv")).read().splitlines()
            self.assertEqual([l for l in lines if l.startswith("# fixture\t")], [want], name)
            self.assertIn("# model\t" + os.path.join(self.folder, "model-00001-of-00002.safetensors"), lines, name)

    def test_logits_layer_outputs_and_topk_equal_the_vendor_model(self):
        for name in ("a", "b", "c", "d", "e"):
            ids = read_ids(self.ids, name)
            logits, outs = self.ref.forward(ids)
            row = f32(self.set(name), "result_output")
            np.testing.assert_allclose(row, logits[-1].numpy(), rtol=2e-4, atol=2e-4, err_msg=name)
            top = [l.split("\t") for l in open(os.path.join(self.set(name), "topk.tsv")).read().splitlines()]
            self.assertEqual(int(top[0][1]), int(logits[-1].argmax()), name)
            self.assertEqual(len(top), 40 if VOCAB >= 40 else VOCAB)
            if name in ("b", "c"):
                for L in range(LAYERS):
                    got = f32(self.set(name), "l_out-%d" % L)
                    want = outs[L][0].numpy() if L < LAYERS - 1 else outs[L][0, -1:].numpy()
                    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=2e-4, atol=2e-4, err_msg="%s layer %d" % (name, L))

    def test_greedy_tokens_equal_the_vendor_models(self):
        for name in ("a", "b", "c"):
            ids = read_ids(self.ids, name)
            rows = [l.split("\t") for l in open(os.path.join(self.set(name), "greedy.tsv")).read().splitlines() if not l.startswith("#")]
            self.assertEqual([int(r[0]) for r in rows], list(range(6)))
            self.assertEqual([int(r[1]) for r in rows], self.ref.greedy(ids, 6), name)
            self.assertTrue(all(float(r[2]) >= 0 for r in rows))

    def test_sets_without_a_continuation_say_so(self):
        for name in ("d", "e", "e_perhead"):
            text = open(os.path.join(self.set(name), "MANIFEST.tsv")).read()
            self.assertIn("# greedy\tgreedy.tsv\t0\n", text)
            self.assertEqual(open(os.path.join(self.set(name), "greedy.tsv")).read(), "# step\tid\tmargin\tlogprob\n")

    def test_the_manifest_rows_have_the_widths_their_columns_state(self):
        for name in ("a", "e"):
            lines = open(os.path.join(self.set(name), "MANIFEST.tsv")).read().splitlines()
            kind = next(l for l in lines if l.startswith("# kind\t")).split("\t")[1:]
            for l in lines:
                if l.startswith("tensor\t"):
                    self.assertEqual(len(l.split("\t")), len(kind) + 1, l)
                    f = l.split("\t")
                    n = int(np.prod([int(x) for x in f[4:8]]))
                    self.assertEqual(int(f[8]), 4 * n, l)
                    self.assertEqual(os.path.getsize(os.path.join(self.set(name), "%s.%s.f32" % (f[1], f[2]))), 4 * n, l)

    def test_the_taps_of_set_a_are_the_graphs_nodes(self):
        a = self.set("a")
        ids = read_ids(self.ids, "a")
        T = len(ids)
        rows = [l.split("\t") for l in open(os.path.join(a, "MANIFEST.tsv")).read().splitlines() if l.startswith("tensor\t")]
        by = {(r[1], int(r[2])): r for r in rows}
        nes = lambda key: [int(x) for x in by[key][4:8]]
        self.assertEqual(nes(("attn_norm-0", 0)), [HID, T, 1, 1])
        self.assertEqual([by[("Qcur-0", i)][10] for i in range(3)], ["MUL_MAT", "RESHAPE", "ROPE"])
        self.assertEqual(nes(("Qcur-0", 0)), [HEADS * HD, T, 1, 1])
        self.assertEqual(nes(("Qcur-0", 2)), [HD, HEADS, T, 1])
        self.assertEqual(nes(("Kcur_normed-0", 0)), [HD, KV, T, 1])
        self.assertEqual(nes(("msa_bs-0", 0)), [64, IDX_HEADS, T, 1])
        self.assertEqual(nes(("msa_block_mask-0", 0)), [64, T, IDX_HEADS, 1])
        self.assertEqual(by[("msa_block_mask-0", 0)][3], "f16")
        self.assertEqual(nes(("msa_fattn-0", 0)), [HD, HEADS // KV, T, KV])
        self.assertEqual(nes(("kqv_out-0", 0)), [HEADS * HD, T, 1, 1])
        self.assertEqual(nes(("ffn_moe_probs-1", 0)), [EXPERTS, T, 1, 1])
        self.assertEqual(nes(("ffn_moe_topk-1", 0)), [TOPK, T, 1, 1])
        self.assertEqual(nes(("ffn_moe_weights_scaled-1", 0)), [1, TOPK, T, 1])
        self.assertNotIn(("ffn_moe_topk-0", 0), by)  # the dense layer has no router
        self.assertEqual(by[("ffn_out-0", 0)][3], "f32")
        # the last layer holds the last position alone after attention, and result_norm too
        self.assertEqual(nes(("l_out-2", 0)), [HID, 1, 1, 1])
        self.assertEqual(nes(("ffn_moe_topk-2", 0)), [TOPK, 1, 1, 1])
        self.assertEqual(nes(("kqv_out-2", 0)), [HEADS * HD, T, 1, 1])
        self.assertEqual(nes(("result_norm", 0)), [HID, 1, 1, 1])
        self.assertEqual(nes(("result_output", 0)), [VOCAB, 1, 1, 1])

    def test_routes_are_ordered_by_the_selection_score_and_match_the_vendor_router(self):
        a = self.set("a")
        ids = read_ids(self.ids, "a")
        T = len(ids)
        topk = np.fromfile(os.path.join(a, "ffn_moe_topk-1.0.i32"), dtype="<i4").reshape(T, TOPK)
        probs = f32(a, "ffn_moe_probs-1").reshape(T, EXPERTS)
        bias = safe_open(os.path.join(self.folder, "model-00002-of-00002.safetensors"), "pt").get_tensor(PFX + "layers.1.block_sparse_moe.e_score_correction_bias").numpy()
        sel = probs + bias
        for t in range(T):
            want = np.lexsort((np.arange(EXPERTS), -sel[t]))[:TOPK]
            self.assertEqual(topk[t].tolist(), want.tolist(), t)
        weights = f32(a, "ffn_moe_weights_scaled-1").reshape(T, TOPK)
        np.testing.assert_allclose(weights.sum(axis=1), 2.0, rtol=1e-5)
        logical = np.fromfile(os.path.join(a, "ffn_moe_topk-1.0.logical.i32"), dtype="<i4")
        self.assertEqual(logical.tolist(), topk.reshape(-1).tolist())

    def test_msa_scores_are_the_per_head_block_maxima_of_the_indexer(self):
        """msa_bs against a numpy recomputation from the attn_norm tap and the folder's weights."""
        a = self.set("a")
        ids = read_ids(self.ids, "a")
        T = len(ids)
        st = dequantized_state(self.folder, self.ref.cfg)
        x = torch.from_numpy(f32(a, "attn_norm-1").reshape(T, HID))
        hf = dump_tf.hf
        norm = lambda w, t: t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w)
        q = norm(st["model.layers.1.self_attn.indexer.q_norm.weight"], (x @ st["model.layers.1.self_attn.indexer.q_proj.weight"].T).view(T, IDX_HEADS, IDX_DIM))
        k = norm(st["model.layers.1.self_attn.indexer.k_norm.weight"], (x @ st["model.layers.1.self_attn.indexer.k_proj.weight"].T).view(T, 1, IDX_DIM))
        rot = hf.MiniMaxM3VLRotaryEmbedding(config=self.ref.cfg)
        cos, sin = rot(x[None], position_ids=torch.arange(T)[None])
        q, k = dump_tf.ORIG_ROPE(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[..., :IDX_DIM], sin[..., :IDX_DIM])
        scores = np.einsum("hqd,sd->hqs", q[0].numpy(), k[0, 0].numpy())
        got = f32(a, "msa_bs-1").reshape(T, IDX_HEADS, 64)
        for h in range(IDX_HEADS):
            for t in range(T):
                for b in range(-(-T // BLOCK)):
                    visible = [s for s in range(b * BLOCK, min((b + 1) * BLOCK, T)) if s <= t]
                    want = max(scores[h, t, s] for s in visible) if visible else -np.finfo(np.float32).max
                    self.assertAlmostEqual(float(got[t, h, b]), float(want), delta=2e-5 * max(1.0, abs(float(want))) if visible else 0, msg=(h, t, b))
        bsf = f32(a, "msa_bsf-1").reshape(T, IDX_HEADS, 64)
        for t in range(T):
            own = t // BLOCK
            self.assertGreaterEqual(float(bsf[t, 0, own]), 1e29)
            self.assertAlmostEqual(float(bsf[t, 0, (own + 1) % 64 if own + 1 < -(-T // BLOCK) else 63]) < 1e29, True)

    def test_the_two_rules_pick_the_same_blocks_while_every_visible_block_fits_and_differ_past_it(self):
        pooled = f32(self.set("e"), "msa_block_mask-1")
        head = f32(self.set("e_perhead"), "msa_block_mask-1")
        T = PROMPTS["e"]
        pooled, head = pooled.reshape(IDX_HEADS, T, 64), head.reshape(IDX_HEADS, T, 64)
        # the vendor's rule gives every head the same blocks
        self.assertTrue(np.array_equal(pooled[0], pooled[1]))
        picked = lambda m, h, t: int((m[h, t] == 0).sum())
        for t in range(T):
            visible = t // BLOCK + 1
            self.assertEqual(picked(pooled, 0, t), min(TOPK_BLOCKS, visible), t)
            self.assertEqual(picked(head, 0, t), min(TOPK_BLOCKS, visible), t)
            if visible <= TOPK_BLOCKS:
                self.assertTrue(np.array_equal(pooled[:, t], head[:, t]), t)
        # past 3 visible blocks the per-head rule differs from the pooled one somewhere, in some layer
        differ = False
        for L in range(LAYERS):
            p = f32(self.set("e"), "msa_block_mask-%d" % L).reshape(IDX_HEADS, T, 64)
            h = f32(self.set("e_perhead"), "msa_block_mask-%d" % L).reshape(IDX_HEADS, T, 64)
            differ |= not np.array_equal(p, h)
        self.assertTrue(differ)
        # the picked blocks always include the token's own and never a hidden one
        for t in range(T):
            self.assertEqual(pooled[0, t, t // BLOCK], 0.0)
            self.assertTrue(np.all(pooled[0, t, t // BLOCK + 1:] == -np.inf))

    def test_the_vendor_rule_matches_the_vendor_model_and_the_per_head_rule_matches_an_independent_one(self):
        ids = read_ids(self.ids, "e")
        vendor_logits, _ = self.ref.forward(ids)
        np.testing.assert_allclose(f32(self.set("e"), "result_output"), vendor_logits[-1].numpy(), rtol=2e-4, atol=2e-4)
        # llama.cpp's rule, written independently here, patched into the vendor model
        hf = dump_tf.hf
        patched = Reference.__new__(Reference)
        patched.__dict__.update(self.ref.__dict__)

        def per_head_forward(mod, hidden_states, position_embeddings, past_key_values, position_ids):
            B, S, _ = hidden_states.shape
            q = mod.q_norm(mod.q_proj(hidden_states).view(B, S, -1, mod.head_dim)).transpose(1, 2)
            k = mod.k_norm(mod.k_proj(hidden_states).view(B, S, 1, mod.head_dim)).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = dump_tf.ORIG_ROPE(q, k, cos[..., :mod.head_dim], sin[..., :mod.head_dim])
            sc = np.einsum("bhqd,bkd->bhqk", q.numpy(), k[:, 0].numpy())
            nb = -(-S // mod.block_size)
            picks = np.full((B, mod.num_heads, S, mod.topk_blocks), -1, dtype=np.int64)
            for h in range(mod.num_heads):
                for t in range(S):
                    best = []
                    for b in range(nb):
                        vis = [s for s in range(b * mod.block_size, min((b + 1) * mod.block_size, S)) if s <= t]
                        if vis:
                            best.append((float("inf") if b == t // mod.block_size else float(max(sc[0, h, t, s] for s in vis)), -b))
                    best.sort(reverse=True)
                    for j, (_, nb_) in enumerate(best[: mod.topk_blocks]):
                        picks[0, h, t, j] = -nb_
            return torch.from_numpy(picks)

        def per_head_mask(mod, block_indices, attention_mask, key_length, dtype, device, position_ids):
            B, H, S, _ = block_indices.shape
            keep = torch.zeros(B, H, S, key_length, dtype=torch.bool)
            for h in range(H):
                for t in range(S):
                    for b in block_indices[0, h, t].tolist():
                        if b >= 0:
                            keep[0, h, t, b * mod.block_size:(b + 1) * mod.block_size] = True
            keep &= torch.arange(key_length)[None, None, None, :] <= torch.arange(S)[None, None, :, None]
            m = torch.zeros(keep.shape, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
            return m.repeat_interleave(HEADS // H, dim=1)

        saved = []
        for layer in patched.model.model.layers:
            ix = layer.self_attn.indexer
            saved.append((ix, ix.forward, ix.build_block_mask))
            ix.forward = lambda *a, ix=ix: per_head_forward(ix, *a)
            ix.build_block_mask = lambda *a, ix=ix: per_head_mask(ix, *a)
        try:
            other_logits, _ = patched.forward(ids)
        finally:
            for ix, f, b in saved:
                ix.forward, ix.build_block_mask = f, b
        np.testing.assert_allclose(f32(self.set("e_perhead"), "result_output"), other_logits[-1].numpy(), rtol=2e-4, atol=2e-4)
        # the rules are not the same function: past 3 blocks the logits differ
        self.assertGreater(float(np.abs(other_logits[-1].numpy() - vendor_logits[-1].numpy()).max()), 1e-3)

    def test_the_ffn_taps_add_up(self):
        """ffn_out is the routed output (scaled) plus the shared one, l_out the residual plus ffn_out: the identities a
        dropped routed_scaling_factor or a wrong residual in a tap would break."""
        a = self.set("a")
        T = PROMPTS["a"]
        for L in (1, 2):
            rows = 1 if L == LAYERS - 1 else T
            moe, shexp = f32(a, "ffn_moe_out-%d" % L), f32(a, "ffn_shexp-%d" % L)
            self.assertTrue(np.array_equal(moe + shexp, f32(a, "ffn_out-%d" % L)), L)
            self.assertEqual(moe.size, rows * HID)
        for L in range(LAYERS):
            inp, out, l_out = f32(a, "ffn_inp-%d" % L), f32(a, "ffn_out-%d" % L), f32(a, "l_out-%d" % L)
            self.assertTrue(np.array_equal(inp + out, l_out), L)

    def test_each_set_holds_the_taps_its_spec_names(self):
        def names(set_name):
            lines = open(os.path.join(self.set(set_name), "MANIFEST.tsv")).read().splitlines()
            return {l.split("\t")[1].rsplit("-", 1)[0] if l.split("\t")[1][-1:].isdigit() else l.split("\t")[1] for l in lines if l.startswith("tensor\t")}

        self.assertEqual(names("a"), set(dump_tf.ALL_TAPS) | {"result_output"})
        self.assertEqual(names("b"), {"l_out", "ffn_moe_topk", "result_output"})
        self.assertEqual(names("c"), {"l_out", "ffn_moe_topk", "result_output"})
        self.assertEqual(names("d"), {"result_output"})
        self.assertEqual(names("e"), set(dump_tf.MSA_TAPS) | {"result_output"})
        self.assertEqual(names("e_perhead"), set(dump_tf.MSA_TAPS) | {"result_output"})

    def test_the_vendor_router_does_not_order_its_picks_so_the_sort_matters(self):
        """HF's torch.topk(sorted=False) leaves the picks in no score order; the tap sorts them."""
        seen = []
        h = self.ref.model.model.layers[1].mlp.gate.register_forward_hook(lambda m, a, o: seen.append(o))
        try:
            self.ref.forward(read_ids(self.ids, "a"))
        finally:
            h.remove()
        logits, _, index = seen[0]
        sel = torch.sigmoid(logits.float()) + self.ref.model.model.layers[1].mlp.gate.e_score_correction_bias
        order = sel.sort(dim=-1, descending=True).indices[:, :TOPK]
        self.assertFalse(torch.equal(order, index), "the raw picks happen to be sorted: the order test has no teeth")

    def test_result_norm_is_the_final_norm_of_the_last_row(self):
        ids = read_ids(self.ids, "a")
        out = {}
        h = self.ref.model.model.norm.register_forward_hook(lambda m, a, o: out.setdefault("x", o.detach().clone()))
        try:
            self.ref.forward(ids)
        finally:
            h.remove()
        np.testing.assert_allclose(f32(self.set("a"), "result_norm"), out["x"][0, -1].numpy(), rtol=2e-4, atol=2e-4)


class Mutants(unittest.TestCase):
    """Each named change of the dumper must turn the comparison with the vendor model red."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mm31-mut-")
        cls.folder = os.path.join(cls.tmp, "model")
        cls.ids = os.path.join(cls.tmp, "ids")
        make_folder(cls.folder)
        make_ids(cls.ids)
        cls.ref = Reference(cls.folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def last_logits(self, tag, **patch):
        """result_output of set b under patched dumper attributes (restored after)."""
        saved = {k: getattr(dump_tf, k) for k in patch}
        for k, v in patch.items():
            setattr(dump_tf, k, v)
        out = os.path.join(self.tmp, tag)
        try:
            rc = dump(self.folder, self.ids, out, ["--sets", "b", "--phase", "prefill"])
        finally:
            for k, v in saved.items():
                setattr(dump_tf, k, v)
        self.assertEqual(rc, 0)
        return f32(os.path.join(out, "ref_minimaxm31_b"), "result_output")

    def assert_red(self, row):
        logits, _ = self.ref.forward(read_ids(self.ids, "b"))
        err = float(np.abs(row - logits[-1].numpy()).max())
        self.assertGreater(err, 1e-2, "the mutant matched the vendor model (max error %g)" % err)

    def test_clean_run_is_green(self):
        logits, _ = self.ref.forward(read_ids(self.ids, "b"))
        np.testing.assert_allclose(self.last_logits("clean"), logits[-1].numpy(), rtol=2e-4, atol=2e-4)

    def test_mutant_norms_with_the_gguf_plus_one_is_red(self):
        self.assert_red(self.last_logits("norm1", norm_weight=lambda t: t.float() + 1.0))

    def test_mutant_gate_and_up_swapped_is_red(self):
        self.assert_red(self.last_logits("swap", cat_gate_up=lambda gate, up: torch.cat([up, gate], dim=0)))


class Refusals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mm31-ref-")
        cls.ids = os.path.join(cls.tmp, "ids")
        make_ids(cls.ids)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_dump(self, folder, *extra, out="out"):
        rc = dump(folder, self.ids, os.path.join(self.tmp, out), ["--sets", "a", "--phase", "prefill", *extra])
        return rc

    def test_a_folder_that_is_not_the_revision_is_refused_by_name(self):
        folder = os.path.join(self.tmp, "plain")
        make_folder(folder, fixture=False)
        with self.assertRaisesRegex(dump_tf.Refusal, "index sha256"):
            dump_tf.fixture_line(dump_tf.Folder(folder))
        self.assertEqual(self.run_dump(folder, out="o1"), 65)

    def test_a_cut_of_another_revision_is_refused(self):
        folder = os.path.join(self.tmp, "other-rev")
        make_folder(folder)
        path = os.path.join(folder, "model-00001-of-00002.safetensors")
        from safetensors.torch import load_file
        t = load_file(path)
        save_file(t, path, metadata={"format": "pt", "bloomery.fixture.rev": "deadbeef", "bloomery.fixture.source": "synthetic"})
        with self.assertRaisesRegex(dump_tf.Refusal, "a cut of revision"):
            dump_tf.fixture_line(dump_tf.Folder(folder))

    def test_a_cut_without_refsets_line_is_refused_by_name(self):
        folder = os.path.join(self.tmp, "cut-no-line")
        make_folder(folder)
        with self.assertRaisesRegex(dump_tf.Refusal, r"refset-check --fixture-line"):
            dump_tf.fixture_line(dump_tf.Folder(folder))
        self.assertEqual(dump(folder, self.ids, os.path.join(self.tmp, "o-no-line"), ["--sets", "a", "--phase", "prefill"], line=False), 65)

    def test_a_line_that_is_not_this_cuts_is_refused(self):
        folder = os.path.join(self.tmp, "cut-lines")
        make_folder(folder)
        good = cut_line(folder)
        cut = dump_tf.Folder(folder)
        self.assertEqual(dump_tf.fixture_line(cut, good), good[len("# fixture\t"):])
        for bad, why in (
            (good.replace("ff=%d" % INTER, "ff=999"), "bloomery.fixture.ff=999"),
            (good.replace("# fixture\t", "# fixtur\t"), "not one"),
            (good + " bloomery.fixture.ff=%d" % INTER, "once"),
            (good + " layers=0", "key=value"),
            ("# fixture\tbloomery.fixture.rev=deadbeef", "states"),
            (good.replace(" bloomery.fixture.layers=0,1,2", ""), "bloomery.fixture.layers=nothing"),
            ("# fixture\tbloomery.fixture.rev=" + dump_tf.REV, "states"),
            (good.replace(dump_tf.REV, "deadbeef"), "bloomery.fixture.rev=deadbeef"),
            (good + " bloomery.fixture.card_budget=30000000000", "bloomery.fixture.card_budget=30000000000"),
        ):
            with self.assertRaisesRegex(dump_tf.Refusal, why, msg=bad):
                dump_tf.fixture_line(cut, bad)
        rc = dump(folder, self.ids, os.path.join(self.tmp, "o-bad-line"), ["--sets", "a", "--phase", "prefill", "--fixture-line", good.replace("ff=%d" % INTER, "ff=999")], line=False)
        self.assertEqual(rc, 65)
        # a line naming only the revision would leave a resumed run's identity equal across a regenerated cut
        rc = dump(folder, self.ids, os.path.join(self.tmp, "o-partial-line"), ["--sets", "a", "--phase", "prefill", "--fixture-line", "# fixture\tbloomery.fixture.rev=" + dump_tf.REV], line=False)
        self.assertEqual(rc, 65)

    def test_kind_names_the_vendors_folder_and_a_cut_and_refuses_the_rest(self):
        import contextlib
        import io
        from unittest import mock

        cut, plain = os.path.join(self.tmp, "kind-cut"), os.path.join(self.tmp, "kind-plain")
        make_folder(cut)
        make_folder(plain, fixture=False)

        def kind(folder):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = dump_tf.main(["--kind", "--folder", folder])
            return rc, out.getvalue().strip()

        self.assertEqual(kind(cut), (0, "cut"))
        self.assertEqual(kind(plain)[0], 65)
        with mock.patch.object(dump_tf, "INDEX_SHA256", dump_tf.Folder(plain).index_sha256):
            self.assertEqual(kind(plain), (0, "folder"))

    def test_the_dequantization_check_runs_on_a_cut_without_its_line(self):
        cut = os.path.join(self.tmp, "selftest-cut")
        make_folder(cut)
        self.assertEqual(dump_tf.main(["--dequant-selftest", "--folder", cut, "--ids", self.ids]), 0)

    def test_a_line_for_the_vendors_folder_is_refused(self):
        folder = os.path.join(self.tmp, "plain-line")
        make_folder(folder, fixture=False)
        from unittest import mock

        with mock.patch.object(dump_tf, "INDEX_SHA256", dump_tf.Folder(folder).index_sha256):
            with self.assertRaisesRegex(dump_tf.Refusal, "vendor's own folder has no fixture line"):
                dump_tf.fixture_line(dump_tf.Folder(folder), "# fixture\tbloomery.fixture.rev=" + dump_tf.REV)

    def test_an_out_directory_that_holds_something_is_refused(self):
        folder = os.path.join(self.tmp, "ok")
        make_folder(folder)
        out = os.path.join(self.tmp, "busy")
        os.makedirs(out)
        open(os.path.join(out, "x"), "w").close()
        self.assertEqual(self.run_dump(folder, out="busy"), 65)

    def test_a_non_finite_input_global_scale_is_refused(self):
        folder = os.path.join(self.tmp, "nan-igs")
        t = make_folder(folder)
        path = os.path.join(folder, "model-00002-of-00002.safetensors")
        from safetensors.torch import load_file
        data = load_file(path)
        key = PFX + "layers.1.block_sparse_moe.experts.3.w2.input_global_scale"
        data[key] = torch.tensor(float("nan"))
        save_file(data, path, metadata={"format": "pt"})
        self.assertEqual(self.run_dump(folder, out="o3"), 65)

    def test_a_tensor_of_the_wrong_dtype_is_refused(self):
        folder = os.path.join(self.tmp, "bad-dtype")
        make_folder(folder)
        path = os.path.join(folder, "model-00001-of-00002.safetensors")
        from safetensors.torch import load_file
        data = load_file(path)
        key = PFX + "layers.0.input_layernorm.weight"
        data[key] = data[key].float()
        save_file(data, path, metadata={"format": "pt", "bloomery.fixture.rev": dump_tf.REV, "bloomery.fixture.source": "synthetic"})
        self.assertEqual(self.run_dump(folder, out="o4"), 65)

    def test_resume_refuses_state_of_another_writer_or_sets(self):
        folder = os.path.join(self.tmp, "resume")
        make_folder(folder)
        out = os.path.join(self.tmp, "resume-out")
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a", "--phase", "prefill", "--until", "1"]), 0)
        # another set list: the identity differs
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a,b", "--phase", "prefill", "--resume"]), 65)
        # the same identity continues, and ends where an uninterrupted run does
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a", "--phase", "prefill", "--resume"]), 0)
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a", "--phase", "greedy"]), 0)
        whole = os.path.join(self.tmp, "resume-whole")
        self.assertEqual(dump(folder, self.ids, whole, ["--sets", "a"]), 0)
        for name in sorted(os.listdir(os.path.join(out, "ref_minimaxm31_a"))):
            if name == "MANIFEST.tsv":
                continue
            self.assertEqual(
                open(os.path.join(out, "ref_minimaxm31_a", name), "rb").read(),
                open(os.path.join(whole, "ref_minimaxm31_a", name), "rb").read(),
                name,
            )

    def test_resume_refuses_state_of_another_cut_generation(self):
        folder = os.path.join(self.tmp, "resume-cut")
        make_folder(folder)
        out = os.path.join(self.tmp, "resume-cut-out")
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a", "--phase", "prefill", "--until", "1"]), 0)
        # the cut regenerated from other layers: the same tensors, index and config, another fixture line
        from safetensors.torch import load_file

        path = os.path.join(folder, "model-00001-of-00002.safetensors")
        before = dump_tf.Folder(folder).index_sha256
        save_file(load_file(path), path, metadata={"format": "pt", **CUT_KEYS, "bloomery.fixture.layers": "0,1,3"})
        self.assertEqual(dump_tf.Folder(folder).index_sha256, before)
        self.assertEqual(dump(folder, self.ids, out, ["--sets", "a", "--phase", "prefill", "--resume"]), 65)

    def test_greedy_without_a_prefill_is_refused(self):
        folder = os.path.join(self.tmp, "no-prefill")
        make_folder(folder)
        self.assertEqual(dump(folder, self.ids, os.path.join(self.tmp, "np-out"), ["--sets", "a", "--phase", "greedy"]), 65)

    def test_a_usage_error_exits_64(self):
        with self.assertRaises(SystemExit) as cm:
            dump_tf.main(["--folder", "x"])
        self.assertEqual(cm.exception.code, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
