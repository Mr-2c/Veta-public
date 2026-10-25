#!/usr/bin/env python3
"""
gate2_clean.py -- case-fold collision gate (AUTHORITATIVE variant).
CLAUSON multi-GPU source gate.

Fortran is case-INSENSITIVE, so two identifiers that differ only in case are the
SAME symbol. Within one program unit (subroutine/function/module/program scope)
this gate reports any identifier that appears in TWO OR MORE distinct spellings
(e.g. a host variable `A` and a contained dummy `a`, or `amp` vs `Amp`). It
strips comments AND string literals first (so 'ALL TESTS PASSED' does not
false-flag against the all() intrinsic -- that is the FP gate23.py deliberately
keeps). Per-scope, string-stripped => authoritative: reconcile against gate23.

Usage:  tools/gate2_clean.py FILE.F90 [FILE2.F90 ...]
Exit:   0 if no collisions, 1 otherwise.

NOTE: a flagged collision is not always a bug -- two case-spellings can be
intentional/harmless. The gate surfaces them for a human decision (rename one,
typically amplitude -> amp). Treat as a prompt, not an automatic reject.
"""
import re
import sys

IDENT = re.compile(r"[A-Za-z]\w*")
# Fortran keywords/intrinsics that legitimately recur and are not user symbols.
KEYWORDS = {
    "module", "subroutine", "function", "program", "end", "endif", "enddo",
    "do", "if", "then", "else", "elseif", "endwhere", "where", "while",
    "use", "only", "implicit", "none", "contains", "call", "return", "stop",
    "integer", "real", "complex", "logical", "character", "double", "precision",
    "type", "class", "parameter", "allocatable", "pointer", "target", "save",
    "intent", "in", "out", "inout", "dimension", "device", "managed", "host",
    "public", "private", "pure", "elemental", "recursive", "impure", "result",
    "allocate", "deallocate", "nullify", "present", "kind", "len", "size",
    "do_", "go", "to", "goto", "select", "case", "default", "exit", "cycle",
    "and", "or", "not", "eqv", "neqv", "true", "false", "data", "common",
    "merge", "real32", "real64", "int32", "abs", "max", "min", "sum", "all",
    "any", "maxval", "minval", "mod", "exp", "sqrt", "cmplx", "aimag", "conjg",
    "real_", "block", "associate", "interface", "procedure", "optional",
    "print", "write", "read", "format", "open", "close", "error",
}


def strip_and_join(text, keep_strings=False):
    logical, pending = [], ""
    for raw in text.splitlines():
        out, q, i = [], None, 0
        while i < len(raw):
            c = raw[i]
            if q:
                if keep_strings:
                    out.append(c)
                if c == q:
                    q = None
            else:
                if c in ("'", '"'):
                    q = c
                    if keep_strings:
                        out.append(c)
                elif c == "!":
                    break
                else:
                    out.append(c)
            i += 1
        line = "".join(out).rstrip()
        s = (pending + " " + line).strip() if pending else line
        if s.endswith("&"):
            pending = s[:-1].rstrip()
            continue
        pending = ""
        logical.append(re.sub(r"^&\s*", "", s))
    if pending:
        logical.append(pending)
    return logical


def scope_open(st):
    low = st.lower()
    m = re.match(r"^(?:(?:pure|elemental|recursive|impure|module)\s+)*"
                 r"subroutine\s+(\w+)", low)
    if m:
        return ("subroutine", m.group(1))
    if re.search(r"\bfunction\s+(\w+)\s*\(", low):
        m = re.search(r"\bfunction\s+(\w+)", low)
        return ("function", m.group(1))
    m = re.match(r"^module\s+(?!procedure\b)(\w+)", low)
    if m:
        return ("module", m.group(1))
    m = re.match(r"^program\s+(\w+)", low)
    if m:
        return ("program", m.group(1))
    return None


def scope_close(st):
    return re.match(r"^end\s*(subroutine|function|module|program)\b", st.lower())


def collisions_per_scope(lines):
    stack = [("file", "<top>")]
    tokens = {id(stack[-1]): {}}
    order = [stack[-1]]
    findings = []

    def record(scope, lines_tokens):
        key = id(scope)
        d = tokens.setdefault(key, {})
        for tok in lines_tokens:
            d.setdefault(tok.lower(), set()).add(tok)

    for st in lines:
        if not st.strip():
            continue
        op = scope_open(st)
        if op:
            stack.append(op)
            order.append(op)
            record(op, IDENT.findall(st))
            continue
        record(stack[-1], IDENT.findall(st))
        if scope_close(st) and len(stack) > 1:
            stack.pop()

    for scope in order:
        d = tokens.get(id(scope), {})
        for low, spell in d.items():
            if low in KEYWORDS:
                continue
            if len(spell) > 1:
                findings.append((scope, sorted(spell)))
    return findings


def check_file(path):
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"[FAIL] {path}: {e}")
        return False
    findings = collisions_per_scope(strip_and_join(text, keep_strings=False))
    if not findings:
        print(f"[PASS] {path}  (no case-fold collisions)")
        return True
    print(f"[FAIL] {path}  ({len(findings)} case-fold collision(s)):")
    for (kind, name), spell in findings:
        print(f"        {kind} {name}: {{{', '.join(spell)}}}")
    return False


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    allok = True
    for p in argv[1:]:
        allok = check_file(p) and allok
    print("NO COLLISIONS" if allok else "CASE-FOLD COLLISIONS (reconcile vs gate23)")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
