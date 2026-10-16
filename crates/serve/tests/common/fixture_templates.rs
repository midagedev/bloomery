/// Every fixture chat template, by name and source (`<name>-chat-template
/// .jinja` under `fixtures/`, relative to this file): `include!`d by the test
/// targets that pin a template's property, whose tables iterate this list and
/// look their expectation up by name, so a fixture missing from a table fails
/// it by name.
const FIXTURE_TEMPLATES: &[(&str, &str)] = &[
    ("qwen3", include_str!("../fixtures/qwen3-chat-template.jinja")),
    ("qwen36", include_str!("../fixtures/qwen36-chat-template.jinja")),
    ("qwen38", include_str!("../fixtures/qwen38-chat-template.jinja")),
    ("glm5", include_str!("../fixtures/glm5-chat-template.jinja")),
    ("v41", include_str!("../fixtures/v41-chat-template.jinja")),
    ("nemotron", include_str!("../fixtures/nemotron-chat-template.jinja")),
    ("nemotron-bolt", include_str!("../fixtures/nemotron-bolt-chat-template.jinja")),
];
