#include "ref_io.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int fails = 0;

static void expect(bool ok, const char * what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++fails;
}

static bool rejected(const char * flag, const char * value, long long lo, long long hi) {
    long long n = 123;
    const bool ok = parse_i64("ref", flag, value, lo, hi, n);
    return !ok && n == 123;
}

static bool accepted(const char * flag, const char * value, long long lo, long long hi, long long want) {
    long long n = -1;
    return parse_i64("ref", flag, value, lo, hi, n) && n == want;
}

int main() {
    int n32 = 123;
    expect(!parse_i32("ref", "-c", "garbage", 0, n32) && n32 == 123, "i32 rejects garbage");
    expect(parse_i32("ref", "-c", "16", 0, n32) && n32 == 16, "i32 reads 16");
    expect(rejected("-c", "garbage", 0, INT_MAX), "nonnumeric -c");
    expect(rejected("-c", "12x", 0, INT_MAX), "trailing junk -c");
    expect(rejected("-c", "12 ", 0, INT_MAX), "trailing space -c");
    expect(rejected("-t", " 8", 0, INT_MAX), "leading space -t");
    expect(rejected("-t", "-1", 0, INT_MAX), "negative -t");
    expect(rejected("-t", "+8", 0, INT_MAX), "plus -t");
    expect(rejected("-ub", "0", 1, INT_MAX), "zero -ub");
    expect(rejected("--tokens-count", "", 1, LLONG_MAX), "empty count");
    expect(accepted("-c", "0", 0, INT_MAX, 0), "zero context");
    expect(accepted("-t", "16", 0, INT_MAX, 16), "threads");
    expect(accepted("-c", "2147483647", 0, INT_MAX, INT_MAX), "int max");
    expect(rejected("-c", "2147483648", 0, INT_MAX), "past int max");

    char dir[] = "/tmp/ref-io-test-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::string ids = std::string(dir) + "/ids";
    {
        std::ofstream o(ids);
        o << "0\n1\r\n2\n";
    }
    std::vector<int32_t> got;
    expect(read_token_file(ids, 3, got) && got.size() == 3 && got[0] == 0 && got[1] == 1 && got[2] == 2,
           "three ids, one with cr");
    got.clear();
    {
        std::ofstream o(ids);
        o << "0\ngarbage\n";
    }
    expect(!read_token_file(ids, 2, got), "garbage token id");
    got.clear();
    {
        std::ofstream o(ids, std::ios::binary);
        o.write("0\n1\0garbage\n", 12);
    }
    expect(!read_token_file(ids, 2, got), "a NUL inside a token record");
    got.clear();
    {
        std::ofstream o(ids);
        o << "1\n";
    }
    expect(!read_token_file(ids, 2, got), "short token file");
    const float row[2] = {1.5f, -0.25f};
    const std::string raw = std::string(dir) + "/row.f32";
    expect(write_raw(raw, row, 2), "write raw");
    {
        std::ifstream in(raw, std::ios::binary);
        float back[2] = {};
        in.read(reinterpret_cast<char *>(back), sizeof(back));
        expect(in.gcount() == (std::streamsize) sizeof(back) && back[0] == 1.5f && back[1] == -0.25f, "read raw back");
    }
    expect(!write_raw(std::string(dir) + "/missing/row.f32", row, 2), "missing directory");
    if (fails) {
        std::fprintf(stderr, "ref_io_test: %d failed\n", fails);
        return 1;
    }
    std::printf("ref_io_test: ok\n");
    return 0;
}
