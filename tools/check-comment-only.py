#!/usr/bin/env python3
"""주석만 바뀌었는지 판정한다(맥에서, 빌드 없이).

    tools/check-comment-only.py <base-ref> <file.rs>...

각 파일의 <base-ref> 판과 작업 트리 판에서 주석을 걷어내고 공백을 지운 뒤 비교한다.
같으면 코드는 한 글자도 안 바뀐 것이므로 비트 불변이 구성상 성립한다 — 주석 다이어트
라운드의 완료 조건. 문자열·raw 문자열·문자 리터럴 안의 `//`는 주석이 아니다.
"""
import subprocess
import sys


def strip(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            depth, i = 1, i + 2
            while i < n and depth:
                if src[i : i + 2] == "/*":
                    depth, i = depth + 1, i + 2
                elif src[i : i + 2] == "*/":
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif c == "r" and i + 1 < n and src[i + 1] in '#"' and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            j = i + 1
            while j < n and src[j] == "#":
                j += 1
            if j < n and src[j] == '"':
                close = '"' + "#" * (j - i - 1)
                k = src.find(close, j + 1)
                k = n if k < 0 else k + len(close)
                out.append(src[i:k])
                i = k
            else:
                out.append(c)
                i += 1
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif c == "'":
            # char literal ('x', '\n', '\'') vs lifetime ('a): a literal closes within 2-12 chars.
            if src[i + 1 : i + 2] == "\\":
                j = src.find("'", i + 3)
                out.append(src[i : j + 1])
                i = j + 1
            elif src[i + 2 : i + 3] == "'":
                out.append(src[i : i + 3])
                i += 3
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join("".join(out).split())


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    base, files = sys.argv[1], sys.argv[2:]
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    bad = 0
    for f in files:
        rel = subprocess.check_output(["git", "ls-files", "--full-name", f], text=True).strip() or f
        old = subprocess.check_output(["git", "show", f"{base}:{rel}"], text=True, cwd=top)
        new = open(f, encoding="utf-8").read()
        a, b = strip(old), strip(new)
        if a == b:
            print(f"comment-only: {rel}")
        else:
            k = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
            print(f"CODE CHANGED: {rel}\n  base: …{a[max(0, k - 60) : k + 60]}…\n  now:  …{b[max(0, k - 60) : k + 60]}…", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
