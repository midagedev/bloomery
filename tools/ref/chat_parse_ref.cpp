// chat_parse_ref — what llama.cpp mainline's chat layer makes of one generation: the template's
// prompt and parser for one user turn, then the generated text parsed as llama-server parses it.
//
//   chat_parse_ref --template <jinja> --output <text file> [--tools <json file>] [--thinking on|off]
//                  [--user <text>] [--reasoning-format auto|none]
//
// --template is a chat template's source; --output is the model's generated text, byte for byte
// (no newline is added or removed). --tools is an OpenAI `tools` array; without it the request
// carries no tools. --thinking sets `enable_thinking` (default on). --reasoning-format is the
// request's `reasoning_format` (default `auto`, llama-server's). No model is opened: the template
// runs with empty BOS and EOS.
//
// Prints one JSON object on stdout: the chat format's name, the generation prompt, the prompt's
// last 64 bytes, and the parsed message in OpenAI's shape (`content`, `reasoning_content`,
// `tool_calls`). A template that does not render or an output that does not parse exits 1 with
// mainline's message on stderr.

#include "chat.h"
#include "peg-parser.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static bool read_file(const char * path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "chat_parse_ref: cannot open %s\n", path);
        return false;
    }
    std::ostringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
}

int main(int argc, char ** argv) {
    const char * template_path = nullptr;
    const char * output_path   = nullptr;
    const char * tools_path    = nullptr;
    std::string  user          = "Hello";
    bool         thinking      = true;
    common_reasoning_format reasoning = COMMON_REASONING_FORMAT_AUTO;
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "--template") == 0 && more) {
            template_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && more) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--tools") == 0 && more) {
            tools_path = argv[++i];
        } else if (strcmp(argv[i], "--user") == 0 && more) {
            user = argv[++i];
        } else if (strcmp(argv[i], "--reasoning-format") == 0 && more) {
            const std::string v = argv[++i];
            if (v != "auto" && v != "none") {
                fprintf(stderr, "chat_parse_ref: --reasoning-format wants auto or none, got '%s'\n", v.c_str());
                return 2;
            }
            reasoning = v == "auto" ? COMMON_REASONING_FORMAT_AUTO : COMMON_REASONING_FORMAT_NONE;
        } else if (strcmp(argv[i], "--thinking") == 0 && more) {
            const std::string v = argv[++i];
            if (v != "on" && v != "off") {
                fprintf(stderr, "chat_parse_ref: --thinking wants on or off, got '%s'\n", v.c_str());
                return 2;
            }
            thinking = v == "on";
        } else {
            fprintf(stderr, "chat_parse_ref: unknown or incomplete argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (!template_path || !output_path) {
        fprintf(stderr,
                "usage: chat_parse_ref --template <jinja> --output <text file> [--tools <json file>] "
                "[--thinking on|off] [--user <text>] [--reasoning-format auto|none]\n");
        return 2;
    }
    std::string source, output, tools_text;
    if (!read_file(template_path, source) || !read_file(output_path, output)) {
        return 2;
    }
    if (tools_path && !read_file(tools_path, tools_text)) {
        return 2;
    }
    try {
        auto tmpls = common_chat_templates_init(nullptr, source, "", "");
        common_chat_templates_inputs inputs;
        common_chat_msg              msg;
        msg.role    = "user";
        msg.content = user;
        inputs.messages              = { msg };
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;
        inputs.enable_thinking       = thinking;
        inputs.reasoning_format      = reasoning;
        if (tools_path) {
            inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools_text));
        }
        const common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
        common_chat_parser_params    pp(params);
        pp.reasoning_format = reasoning;
        common_chat_msg parsed;
        if (params.parser.empty()) {
            parsed = common_chat_parse(output, false, pp);
        } else {
            common_peg_arena arena;
            arena.load(params.parser);
            parsed = common_chat_peg_parse(arena, output, false, pp);
        }
        common_json out;
        out["format"]            = std::string(common_chat_format_name(params.format));
        out["generation_prompt"] = params.generation_prompt;
        const size_t n           = params.prompt.size();
        out["prompt_tail"]       = params.prompt.substr(n > 64 ? n - 64 : 0);
        out["message"]           = parsed.to_json_oaicompat();
        printf("%s\n", out.dump().c_str());
    } catch (const std::exception & e) {
        fprintf(stderr, "chat_parse_ref: %s\n", e.what());
        return 1;
    }
    return 0;
}
