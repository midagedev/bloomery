// Gemma4a's tower oracle, consuming the pinned host preprocessor's mel files.
#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#define DUMP_TOOL "dump_audio"
#include "../clefvis/set_writer.h"

struct taps {
    set_writer * out;
    std::string prefix;
    std::set<std::string> seen;
};

static bool observe(ggml_tensor * t, bool ask, void * user) {
    auto & s = *static_cast<taps *>(user);
    std::string name;
    const ggml_tensor * value = t;
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && std::string(t->src[0]->name) == "a.input_projection.weight") name = "sscp";
    if (t->op == GGML_OP_ADD && t->src[1] && std::string(t->src[1]->name) == "a.pre_encode.out.bias") name = "pre_encode";
    if (t->op == GGML_OP_MUL && t->src[1]) {
        for (int b = 0; b < 12; ++b) {
            if (std::string(t->src[1]->name) == "a.blk." + std::to_string(b) + ".ln2.weight") name = "blk" + std::to_string(b);
        }
    }
    if (name.empty()) {
        for (auto * src : t->src) {
            if (!src) continue;
            for (const char * input : {"pos_emb", "kq_mask"}) {
                if (strstr(src->name, input)) {
                    name = input;
                    value = src;
                    break;
                }
            }
            if (!name.empty()) break;
        }
    }
    if (ask) return !name.empty();
    if (!name.empty() && s.seen.insert(name).second) {
        if (value->type != GGML_TYPE_F32 || !ggml_is_contiguous(value)) DIE("%s: non-contiguous F32 tap", name.c_str());
        std::vector<float> data(ggml_nelements(value));
        ggml_backend_tensor_get(value, data.data(), 0, data.size() * sizeof(float));
        s.out->tensor_f32(s.prefix + "/" + name, "NONE", value->ne, data.data());
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6 || (argc - 4) % 2) DIE("usage: dump_audio <mmproj> <new-set-directory> <cpu|gpu> <name> <mel.f32> ...");
    const std::string device(argv[3]);
    if (device != "cpu" && device != "gpu") DIE("device must be cpu or gpu");
    const char * build = getenv("BLOOMERY_REF_BUILD");
    if (!build || strlen(build) != 9 || strspn(build, "0123456789abcdef") != 9) DIE("BLOOMERY_REF_BUILD requires the nine-digit oracle commit");
    ggml_backend_load_all();
    auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (device == "gpu" && !gpu) DIE("gpu mode requires a visible GPU backend");
    std::set<std::string> names;
    for (int a = 4; a < argc; a += 2) {
        const std::string name(argv[a]);
        if (name.empty() || name.size() > 128 || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-") != std::string::npos ||
            !names.insert(safe_name(name)).second) DIE("invalid or colliding case name: %s", argv[a]);
    }
    if (mkdir(argv[2], 0755) != 0) DIE("output path must be new: %s", argv[2]);
    set_writer out(argv[2]);
    taps state{&out, "", {}};
    clip_context_params params{};
    params.use_gpu = device == "gpu";
    params.device = params.use_gpu ? gpu : nullptr;
    params.warmup = false;
    auto clean = clip_init(argv[1], params);
    params.cb_eval = observe;
    params.cb_eval_user_data = &state;
    auto tapped = clip_init(argv[1], params);
    if (!clean.ctx_a || !tapped.ctx_a) DIE("audio projector did not open");
    out.line(std::string("# model\t") + argv[1]);
    out.line("# arch\tgemma4a");
    out.line(std::string("# build\t") + build);
    out.line("# device\t" + device);
    out.line("# tapped-output\tbit-identical");
    for (int a = 4; a < argc; a += 2) {
        const std::string name(argv[a]);
        std::ifstream file(argv[a + 1], std::ios::binary | std::ios::ate);
        if (!file) DIE("cannot open %s", argv[a + 1]);
        const auto bytes = file.tellg();
        if (bytes <= 0 || bytes % (128 * sizeof(float)) || bytes > 2999 * 128 * sizeof(float)) DIE("bad mel file %s", argv[a + 1]);
        const int frames = int(bytes / (128 * sizeof(float)));
        std::vector<float> mel(size_t(frames) * 128);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(mel.data()), bytes)) DIE("short mel read");
        for (float value : mel) if (!std::isfinite(value)) DIE("non-finite mel file %s", argv[a + 1]);
        clip_image_f32 image;
        image.set_size({frames, 128}, false, true);
        image.cpy_buf(mel);
        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(image));
        const int tokens = (frames + 3) / 4;
        std::vector<float> expected(size_t(tokens) * 512), observed(expected.size());
        state.prefix = name;
        state.seen.clear();
        if (!clip_image_batch_encode(clean.ctx_a, 8, &batch, expected) || !clip_image_batch_encode(tapped.ctx_a, 8, &batch, observed)) DIE("%s: encode failed", name.c_str());
        if (state.seen.size() != 16) {
            std::string got;
            for (const auto & key : state.seen) got += " " + key;
            DIE("%s: expected 16 taps, got %zu:%s", name.c_str(), state.seen.size(), got.c_str());
        }
        if (expected.size() != size_t(tokens) * 512 || expected.size() != observed.size() || memcmp(expected.data(), observed.data(), expected.size() * sizeof(float))) DIE("%s: tapped output differs", name.c_str());
        const int64_t mn[4] = {frames, 128, 1, 1};
        const int64_t en[4] = {512, tokens, 1, 1};
        out.tensor_f32(name + "/mel", "NONE", mn, mel.data());
        out.tensor_f32(name + "/embd", "NONE", en, expected.data());
        fprintf(stderr, "dump_audio: %s %s frames=%d tokens=%d taps=%zu tapped=bit-identical\n", device.c_str(), name.c_str(), frames, tokens, state.seen.size());
    }
    out.finish();
    clip_free(clean.ctx_v); clip_free(clean.ctx_a); clip_free(clean.ctx_gen_a);
    clip_free(tapped.ctx_v); clip_free(tapped.ctx_a); clip_free(tapped.ctx_gen_a);
}
