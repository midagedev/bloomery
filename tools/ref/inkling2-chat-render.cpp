// inkling2-chat-render — the reference's own rendering of a chat request through the Inkling 2 template: llama.cpp's
// common_chat_templates_apply (its minja engine and the Inkling hooks of common/parsers/inkling.cpp) on the template file
// and an OpenAI-style request body. Prints the prompt, then (after a line `=== generation prompt`) the generation prompt.
//   inkling2-chat-render TEMPLATE.jinja REQUEST.json [REASONING_EFFORT_JSON]
// REQUEST.json: {"messages":[...], "tools":[...]?}; the optional third argument is the template's `reasoning_effort` as JSON (\"high\", 0.5).
// Linked against the oracle tree's libraries without rebuilding it (tools/ref/inkling2-chat-render.sh).
#include "chat.h"
#include "json.h"
#include <fstream>
#include <iostream>
#include <sstream>

static std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "inkling2-chat-render: cannot read " << path << "\n"; std::exit(66); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::cerr << "usage: inkling2-chat-render TEMPLATE REQUEST [EFFORT_JSON]\n"; return 64; }
    const std::string tmpl = slurp(argv[1]);
    const auto req = common_json::parse(slurp(argv[2]));
    auto tmpls = common_chat_templates_init(nullptr, tmpl);
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.messages = common_chat_msgs_parse_oaicompat(req.at("messages"));
    if (req.contains("tools")) in.tools = common_chat_tools_parse_oaicompat(req.at("tools"));
    in.add_generation_prompt = true;
    if (argc > 3) in.chat_template_kwargs["reasoning_effort"] = argv[3];
    const auto out = common_chat_templates_apply(tmpls.get(), in);
    std::cout << out.prompt.str() << "\n=== generation prompt\n" << out.generation_prompt;
    return 0;
}
