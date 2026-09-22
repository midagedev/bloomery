// Default paths of the reference harnesses (tools/ref/*_ref.cpp), in one place.
//
// ref_model_path(): the gguf the harnesses read. BLOOMERY_REF_MODEL overrides it — the same
// variable the shell runners in this directory read, with the same default.
// ref_data_dir(): the data directory the gates read. tools/box.sh exports BLOOMERY_DATA; a
// parallel track moves both a harness's input and its dump by setting it.
//
// An empty variable counts as unset, like the runners' ${VAR:-default}.
#pragma once

#include <cstdlib>

inline const char * ref_env_or(const char * name, const char * fallback) {
    const char * v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

inline const char * ref_model_path() {
    return ref_env_or("BLOOMERY_REF_MODEL", "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf");
}

inline const char * ref_data_dir() {
    return ref_env_or("BLOOMERY_DATA", "/root/bloomery-data");
}
