#pragma once
// Token files and raw f32 rows for the mainline oracles (hidden_ref, nemotron_ref, q35mtp_ref).
// A token id is a non-negative 32-bit integer. The header does not include llama.h:
// llama_token is that integer, so a vector of it is a vector of int32_t.

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// `value` is one decimal integer in [lo, hi], the whole argument. Anything else is named on stderr.
inline bool parse_i64(const char * who, const char * flag, const char * value, long long lo, long long hi,
                      long long & out) {
    if (value == nullptr || value[0] < '0' || value[0] > '9') {
        fprintf(stderr, "%s: %s wants an integer from %lld to %lld, got %s\n", who, flag,
                lo, hi, value == nullptr ? "" : value);
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long long n = strtoll(value, &end, 10);
    if (end == value || *end != '\0' || errno != 0 || n < lo || n > hi) {
        fprintf(stderr, "%s: %s wants an integer from %lld to %lld, got %s\n", who, flag, lo, hi, value);
        return false;
    }
    out = n;
    return true;
}

// An int in [lo, INT_MAX]. `lo` is 0 where 0 means the harness default, and 1 where 0 is not a size.
inline bool parse_i32(const char * who, const char * flag, const char * value, int lo, int & dest) {
    long long n = 0;
    if (!parse_i64(who, flag, value, lo, INT_MAX, n)) return false;
    dest = (int) n;
    return true;
}

template <typename Token>
bool read_token_file(const std::string & path, long long count, std::vector<Token> & ids) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "%s: cannot read\n", path.c_str());
        return false;
    }
    std::string line;
    long long lines = 0;
    while ((long long) ids.size() < count && std::getline(in, line)) {
        ++lines;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        char * end = nullptr;
        errno = 0;
        const long id = strtol(line.c_str(), &end, 10);
        // The whole record is the number: a NUL inside the line ends the C string but not the record.
        if (line.empty() || line[0] < '0' || line[0] > '9' || end != line.c_str() + line.size() ||
            errno != 0 || id < 0 || id > INT32_MAX) {
            fprintf(stderr, "%s:%lld: invalid token id '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((Token) id);
    }
    if (!in.bad() && (long long) ids.size() == count) return true;
    fprintf(stderr, "%s: holds %zu token ids, asked for %lld\n", path.c_str(), ids.size(), count);
    return false;
}

// The ids, one decimal id a line (the form read_token_file reads), every line written and the file closed.
template <typename Token>
bool write_ids(const std::string & path, const std::vector<Token> & ids) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        perror(path.c_str());
        return false;
    }
    bool ok = true;
    for (Token id : ids) ok = ok && fprintf(f, "%d\n", (int) id) >= 0;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "%s: short write\n", path.c_str());
        return false;
    }
    return true;
}

// `bytes` bytes to `path`, every byte written and the file closed, or named on stderr.
inline bool write_bytes(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        perror(path.c_str());
        return false;
    }
    const bool ok = fwrite(data, 1, bytes, f) == bytes;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "%s: short write\n", path.c_str());
        return false;
    }
    return true;
}

inline bool write_raw(const std::string & path, const float * data, size_t count) {
    return write_bytes(path, data, count * sizeof(float));
}
