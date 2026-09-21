// kvclear_probe — does what was decoded before llama_kv_cache_clear change the next
// sequence's logits?
//
// The reference set this repository gates against was produced by argmax_ref, which
// separates prompts with llama_kv_cache_clear. Measured: one extra decoded token in the
// previous sequence moves the next prompt's logits by ~1.0 (prompt 24: argmax 1191 at
// 30.222 becomes 245 at 29.231), on the CUDA and the CPU backend alike. This tool is the
// minimal form of that: no tokenizer, no files written, one sequence, arms that differ in
// exactly one thing each.
//
// The arms split two candidates the whole-file runs cannot separate, because both priors
// leave EIGHT cells at positions 0..7:
//   - prompt 5 (8 tokens) then clear then prompt 24  -> clean, byte-identical to fresh
//   - prompt 24 (7 tokens) + 1 generated token then clear then prompt 24 -> dirty
// The only things that differ are the token ids in the cells and where the generation loop
// puts its llama_get_logits_ith(-1). `reads` moves that read without changing a cell;
// `feed_argmax` changes the id without moving a read; the R_SKIP arms drop the clear
// entirely and report the prior sequence's own next-token row.
//
// Every arm reports max |L1 - L0| over the full logit vector against L0, the same sequence
// decoded in a context that has never seen anything else. `self_noread`, `ctl_fresh` and
// `prior_noread` are the controls: they must come back bitwise 0, or the tool is measuring
// its own noise.
//
// Build: tools/ref/build-kvclear.sh
// Run:   $OUT/kvclear_probe -m <gguf> --prompts tools/ref/prompts.tsv --seq 24 --prior 5 \
//          -ngl 0 -c 512 -t 32
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<llama_token> read_prompt(const char * path, int want) {
    std::vector<llama_token> out;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "kvclear_probe: cannot read %s\n", path);
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string id, text, ids;
        if (!std::getline(ss, id, '\t')) continue;
        if (!std::getline(ss, text, '\t')) continue;
        if (!std::getline(ss, ids, '\t')) continue;
        if (atoi(id.c_str()) != want) continue;
        std::stringstream ts(ids);
        std::string tok;
        while (std::getline(ts, tok, ',')) out.push_back((llama_token) atoi(tok.c_str()));
        return out;
    }
    return out;
}

static int g_n_vocab = 0;

// The row order the reference files use: logit descending, id ascending.
static std::vector<int> top5(const std::vector<float> & lg) {
    std::vector<int> idx(g_n_vocab);
    for (int i = 0; i < g_n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) {
        if (lg[a] != lg[b]) return lg[a] > lg[b];
        return a < b;
    });
    idx.resize(5);
    return idx;
}

// --trace: hash every graph node the backend computes, so two arms that disagree can be
// diffed down to the first op that differs. The eval callback makes the scheduler compute
// node by node, which can itself change fusion, so a trace run re-checks that the two arms
// still disagree; if they agree under tracing, the instrument moved the thing measured.
static bool     g_trace_on  = false;
static int      g_trace_dec = 0;
static int      g_trace_node = 0;
static const char * g_trace_tag = "";

static int trace_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!g_trace_on) return 0;
    if (ask) return 1;
    const size_t n = ggml_nbytes(t);
    std::vector<char> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= (unsigned char) buf[i]; h *= 1099511628211ULL; }
    printf("T\t%s\t%d\t%d\t%s\t%s\t%lld,%lld\t%016llx\n", g_trace_tag, g_trace_dec,
           g_trace_node++, ggml_op_name(t->op), t->name,
           (long long) t->ne[0], (long long) t->ne[1], (unsigned long long) h);
    return true;
}

// A read slot. llama_get_logits_ith(-1) calls llama_synchronize internally; g_use_sync
// makes the slot call only llama_synchronize, which is the arm that says whether the
// trigger is in the scheduler's synchronize or in the logits/output bookkeeping above it.
static bool g_use_sync = false;
static void read_slot(llama_context * ctx) {
    if (g_use_sync) llama_synchronize(ctx);
    else            (void) llama_get_logits_ith(ctx, -1);
}

static std::vector<float> snapshot(llama_context * ctx) {
    const float * p = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(p, p + g_n_vocab);
}

// Where llama_get_logits_ith(-1) is called while the prior sequence is decoded. argmax_ref
// uses LAST for a prompt (one read, for the top-5 row) and, with --gen, one more read after
// every generated token. EACH and BEFORE_LAST are the arms that move that read earlier
// without changing a single cell.
enum Reads { RD_NONE, RD_LAST, RD_EACH, RD_BEFORE_LAST };
static const char * reads_name(int r) {
    switch (r) {
        case RD_NONE: return "none";
        case RD_LAST: return "last";
        case RD_EACH: return "each";
        default:      return "before_last";
    }
}

// Step-decode `toks` starting at `pos0`, reading logits where `reads` says.
// Returns false on a decode failure.
static bool step_decode(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, int reads) {
    const size_t n = toks.size();
    for (size_t i = 0; i < n; ++i) {
        llama_token t = toks[i];
        g_trace_dec = (int) i; g_trace_node = 0;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1, pos0 + (int) i, 0))) return false;
        const bool rd = reads == RD_EACH
                     || (reads == RD_LAST        && i + 1 == n)
                     || (reads == RD_BEFORE_LAST && i + 2 == n);
        if (rd) read_slot(ctx);
    }
    return true;
}

static bool batch_decode(llama_context * ctx, std::vector<llama_token> toks, int pos0) {
    return llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size(), pos0, 0)) == 0;
}

enum Reset { R_CLEAR, R_SEQRM_ALL, R_SEQRM_0, R_FRESH, R_NONE, R_SKIP };


int main(int argc, char ** argv) {
    const char *        prompts_path = nullptr;
    int                 seq_id = 24, prior_id = 5;
    bool                trace = false;
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc)   prompts_path = argv[++i];
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc)   seq_id   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prior") == 0 && i + 1 < argc) prior_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trace") == 0) trace = true;
        else passthrough.push_back(argv[i]);
    }
    if (!prompts_path) {
        fprintf(stderr, "kvclear_probe: --prompts <tsv> is required; this tool does not tokenize\n");
        return 2;
    }
    std::vector<llama_token> A = read_prompt(prompts_path, seq_id);
    std::vector<llama_token> B = read_prompt(prompts_path, prior_id);
    if (A.empty() || B.empty()) {
        fprintf(stderr, "kvclear_probe: prompt %d or %d not in %s\n", seq_id, prior_id, prompts_path);
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) return 2;
    params.warmup = false;
    if (trace) { params.cb_eval = trace_cb; params.cb_eval_user_data = nullptr; }

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    if (!init.model || !init.context) {
        fprintf(stderr, "kvclear_probe: failed to load the model\n");
        return 1;
    }
    g_n_vocab = llama_n_vocab(init.model);
    llama_context_params cparams = common_context_params_to_llama(params);

    printf("# kvclear_probe\tmodel=%s\tseq=%d(n=%zu)\tprior=%d(n=%zu)\n",
           params.model.c_str(), seq_id, A.size(), prior_id, B.size());
    printf("# params\tn_ctx=%u\tn_batch=%u\tn_ubatch=%u\tflash_attn=%d\tmla=%d\tn_gpu_layers=%d\tn_threads=%d\ttype_k=%d\ttype_v=%d\n",
           cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, (int) cparams.flash_attn,
           (int) params.mla_attn, params.n_gpu_layers, cparams.n_threads,
           (int) cparams.type_k, (int) cparams.type_v);

    // L0: the sequence in a context that has seen nothing else. Every arm is judged
    // against this, not against the arm before it.
    std::vector<float> L0;
    {
        llama_context * c = llama_init_from_model(init.model, cparams);
        if (!c || !step_decode(c, A, 0, RD_NONE)) { fprintf(stderr, "kvclear_probe: L0 failed\n"); return 1; }
        L0 = snapshot(c);
        llama_free(c);
    }
    std::vector<int> t0 = top5(L0);
    printf("# L0 (fresh context, step prefill)\targmax=%d\ttop5=", t0[0]);
    for (int i = 0; i < 5; ++i) printf("%s%d", i ? "," : "", t0[i]);
    printf("\t");
    for (int i = 0; i < 5; ++i) printf("%s%.6f", i ? "," : "", L0[t0[i]]);
    printf("\n\narm\tprior\treads\textra\treset\ttarget\targmax\tmax_abs_diff\tn_diff\ttop5_ids\ttop5_logits\n");

    // One arm: do the prior work in its own context, reset it, decode A, compare with L0.
    // `prior`: 0 none, 1 A, 2 B. `extra` tokens are fed after the prior sequence the way
    // argmax_ref's generation loop feeds them; the id is A[0] unless `feed_argmax`, which
    // costs a logits read of its own and is the arm that shows the id does not matter.
    // reset R_SKIP stops before the reset: L1 is then the prior's OWN last logit row, which
    // is how an arm asks whether a clear is involved at all.
    auto arm = [&](const char * name, int prior, int reads, int extra, bool extra_read,
                   bool feed_argmax, Reset reset, bool tgt_batch, bool prior_batch,
                   bool use_sync = false) {
        g_use_sync = use_sync;
        llama_context * c = llama_init_from_model(init.model, cparams);
        if (!c) { fprintf(stderr, "kvclear_probe: context alloc failed for %s\n", name); return; }
        const std::vector<llama_token> & P = (prior == 2) ? B : A;
        bool ok = true;
        if (prior != 0) {
            ok = prior_batch ? batch_decode(c, P, 0) : step_decode(c, P, 0, reads);
            int pos = (int) P.size();
            for (int s = 0; s < extra && ok; ++s) {
                llama_token feed = feed_argmax ? (llama_token) top5(snapshot(c))[0] : A[0];
                ok = llama_decode(c, llama_batch_get_one(&feed, 1, pos++, 0)) == 0;
                if (ok && extra_read) read_slot(c);
            }
        }
        if (!ok) { fprintf(stderr, "kvclear_probe: prior decode failed for %s\n", name); llama_free(c); return; }

        std::vector<float> L1;
        const char * reset_label = "clear";
        const char * tgt_label   = tgt_batch ? "batch" : "step";
        if (reset == R_SKIP) {
            reset_label = "-";
            tgt_label   = "-";
            L1 = snapshot(c);          // the prior sequence's own next-token row
            llama_free(c);
        } else {
            switch (reset) {
                case R_CLEAR:     llama_kv_cache_clear(c); break;
                case R_SEQRM_ALL: llama_kv_cache_seq_rm(c, -1, -1, -1); reset_label = "seq_rm(-1,-1,-1)"; break;
                case R_SEQRM_0:   llama_kv_cache_seq_rm(c,  0,  0, -1); reset_label = "seq_rm(0,0,-1)";   break;
                case R_FRESH:     llama_free(c); c = llama_init_from_model(init.model, cparams);
                                  reset_label = "fresh_ctx"; break;
                default:          reset_label = "none"; break;   // R_NONE: the contrast, not a fix
            }
            if (!c) { fprintf(stderr, "kvclear_probe: fresh context failed for %s\n", name); return; }
            int pos0 = (reset == R_NONE && prior != 0) ? (int) P.size() + extra : 0;
            g_trace_on = trace;
            ok = tgt_batch ? batch_decode(c, A, pos0) : step_decode(c, A, pos0, RD_NONE);
            g_trace_on = false;
            if (!ok) { fprintf(stderr, "kvclear_probe: target decode failed for %s\n", name); llama_free(c); return; }
            L1 = snapshot(c);
            llama_free(c);
        }
        g_use_sync = false;

        double mx = 0.0;
        long   nd = 0;
        for (int i = 0; i < g_n_vocab; ++i) {
            double d = std::fabs((double) L1[i] - (double) L0[i]);
            if (d > mx) mx = d;
            if (L1[i] != L0[i]) ++nd;
        }
        std::vector<int> t1 = top5(L1);
        printf("%s\t%s\t%s%s\t%d%s\t%s\t%s\t%d\t%.6g\t%ld\t",
               name, prior == 0 ? "none" : (prior == 2 ? "B" : "A"), reads_name(reads),
               use_sync ? "(sync)" : "",
               extra, extra ? (extra_read ? "r" : "-") : "", reset_label, tgt_label,
               t1[0], mx, nd);
        for (int i = 0; i < 5; ++i) printf("%s%d", i ? "," : "", t1[i]);
        printf("\t");
        for (int i = 0; i < 5; ++i) printf("%s%.6f", i ? "," : "", L1[t1[i]]);
        printf("\n");
        fflush(stdout);
    };

    // Group 1 -- no clear anywhere: is the reset involved at all? These stop after the
    // prior sequence and report ITS next-token row, the same row L0 holds.
    //  name                    prior reads          extra rd  argmax reset   tgtB priorB
    if (trace) {
        // The two arms that disagree with nothing else between them: same cells, same
        // positions, same clear -- one synchronize before the clear versus seven. The
        // eval callback perturbs the computation, so these two are compared with EACH
        // OTHER, never with the untraced L0.
        g_trace_tag = "clean"; arm("trace_prior_noread",    1, RD_NONE, 0, false, false, R_CLEAR, false, false);
        g_trace_tag = "dirty"; arm("trace_prior_read_each", 1, RD_EACH, 0, false, false, R_CLEAR, false, false);
        llama_free(init.context);
        llama_free_model(init.model);
        llama_backend_free();
        return 0;
    }

    arm("self_noread",              1, RD_NONE,        0, false, false, R_SKIP, false, false);
    arm("self_read_each",           1, RD_EACH,        0, false, false, R_SKIP, false, false);
    arm("self_read_before_last",    1, RD_BEFORE_LAST, 0, false, false, R_SKIP, false, false);
    arm("self_read_last",           1, RD_LAST,        0, false, false, R_SKIP, false, false);

    // Group 2 -- the reference tool's shapes, with a clear between the sequences.
    arm("ctl_fresh",                0, RD_NONE,        0, false, false, R_CLEAR, false, false);
    arm("gen0_shape",               1, RD_LAST,        0, false, false, R_CLEAR, false, false);
    arm("gen1_shape",               1, RD_LAST,        1, true,  true,  R_CLEAR, false, false);
    arm("gen1_shape_fixedtok",      1, RD_LAST,        1, true,  false, R_CLEAR, false, false);
    arm("gen1_noread_after_extra",  1, RD_LAST,        1, false, false, R_CLEAR, false, false);
    arm("gen2_shape",               1, RD_LAST,        2, true,  true,  R_CLEAR, false, false);
    arm("prior_noread",             1, RD_NONE,        0, false, false, R_CLEAR, false, false);
    arm("prior_read_each",          1, RD_EACH,        0, false, false, R_CLEAR, false, false);
    arm("prior_read_before_last",   1, RD_BEFORE_LAST, 0, false, false, R_CLEAR, false, false);
    arm("priorB_gen0_shape",        2, RD_LAST,        0, false, false, R_CLEAR, false, false);
    arm("priorB_gen1_shape",        2, RD_LAST,        1, true,  false, R_CLEAR, false, false);

    // Group 3 -- the reset variants, all on the shape that goes wrong.
    arm("gen1_seqrm_all",           1, RD_LAST,        1, true,  true,  R_SEQRM_ALL, false, false);
    arm("gen1_seqrm_0",             1, RD_LAST,        1, true,  true,  R_SEQRM_0,   false, false);
    arm("gen1_freshctx",            1, RD_LAST,        1, true,  true,  R_FRESH,     false, false);
    arm("gen1_noreset",             1, RD_LAST,        1, true,  true,  R_NONE,      false, false);

    // Group 4 -- batch instead of step, on both sides.
    arm("gen1_target_batch",        1, RD_LAST,        1, true,  true,  R_CLEAR, true,  false);
    arm("gen0_target_batch",        1, RD_LAST,        0, false, false, R_CLEAR, true,  false);
    arm("gen1_prior_batch",         1, RD_LAST,        1, true,  true,  R_CLEAR, false, true);
    arm("fresh_batch",              0, RD_NONE,        0, false, false, R_CLEAR, true,  false);

    // Group 5 -- the read slot replaced by a bare llama_synchronize. Same call sites, no
    // logits pointer taken: whatever still flips is below llama_get_logits_ith.
    arm("gen1_shape_sync",          1, RD_LAST,        1, true,  false, R_CLEAR, false, false, true);
    arm("prior_read_each_sync",     1, RD_EACH,        0, false, false, R_CLEAR, false, false, true);
    arm("gen0_shape_sync",          1, RD_LAST,        0, false, false, R_CLEAR, false, false, true);


    // The mechanism arms. llama-build-context.cpp:2735 builds the graph with
    //   is_warming_up = lctx.n_eval == 0 && batch.n_tokens == 1 && batch.token[0] == BOS
    // and the ctor at :58 then uses ALL experts: n_expert_used(warmup ? n_expert : n_expert_used).
    // n_eval is incremented in exactly one place, llama_synchronize (llama.cpp:12574), and only
    // when n_queued_tokens == 1 -- so a read after a multi-token run never raises it, and
    // llama_reset_timings (llama.cpp:13851) puts it back to 0. Each arm below flips ONE
    // conjunct of that predicate and nothing else.
    printf("\n# mechanism arms (each flips one conjunct of is_warming_up)\narm\tflipped\targmax\tmax_abs_diff\ttop5_logits\n");
    auto mech = [&](const char * name, const char * what, int lead_tokens, bool sync_after,
                    bool reset_timings, int batch_prefix) {
        llama_context * c = llama_init_from_model(init.model, cparams);
        if (!c) return;
        for (int i = 0; i < lead_tokens; ++i) {
            llama_token t = A[0];                       // BOS, one token per decode
            if (llama_decode(c, llama_batch_get_one(&t, 1, i, 0))) { llama_free(c); return; }
        }
        if (sync_after)     llama_synchronize(c);        // n_queued_tokens == 1 here -> n_eval++
        if (reset_timings)  llama_reset_timings(c);      // puts n_eval back to 0
        if (lead_tokens)    llama_kv_cache_clear(c);
        bool ok = true;
        if (batch_prefix > 1) {
            std::vector<llama_token> head(A.begin(), A.begin() + batch_prefix);
            std::vector<llama_token> rest(A.begin() + batch_prefix, A.end());
            ok = batch_decode(c, head, 0) && step_decode(c, rest, batch_prefix, RD_NONE);
        } else {
            ok = step_decode(c, A, 0, RD_NONE);
        }
        if (!ok) { llama_free(c); return; }
        std::vector<float> L1 = snapshot(c);
        llama_free(c);
        double mx = 0.0;
        for (int i = 0; i < g_n_vocab; ++i) mx = std::max(mx, std::fabs((double) L1[i] - (double) L0[i]));
        std::vector<int> t1 = top5(L1);
        printf("%s\t%s\t%d\t%.6g\t", name, what, t1[0], mx);
        for (int i = 0; i < 5; ++i) printf("%s%.6f", i ? "," : "", L1[t1[i]]);
        printf("\n");
        fflush(stdout);
    };
    //                        what is flipped                    lead sync reset bprefix
    mech("warm0_baseline",   "nothing (== L0)",                     0, false, false, 0);
    mech("warm1_nosync",     "1 lone decode, NO sync: n_eval=0",    1, false, false, 0);
    mech("warm1_sync",       "1 lone decode + sync: n_eval=1",      1, true,  false, 0);
    mech("warm1_sync_reset", "+ llama_reset_timings: n_eval=0",     1, true,  true,  0);
    mech("prefix2_batch",    "first ubatch 2 tokens, n_eval=0",     0, false, false, 2);

    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return 0;
}
