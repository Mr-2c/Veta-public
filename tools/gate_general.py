#!/usr/bin/env python3
"""
gate_general.py -- structural-balance gate for Fortran .F90 files.
CLAUSON multi-GPU source gate.

Matches OPEN vs CLOSE counts for: module, subroutine, function, do, if-then,
type (derived-type DEFINITION). Comments and string literals are stripped first;
continuation lines (trailing '&') are joined into logical lines, and each logical
line is split into statements on ';' so one-liner nests
(`do; do; do ... end do; end do; end do`) and one-liner loops (`do i=1,n; ...; end do`)
balance correctly.

Usage:  tools/gate_general.py FILE.F90 [FILE2.F90 ...]
Exit:   0 if every file is balanced, 1 otherwise.
"""
import re
import sys


def strip_and_join(text):
    """Strip comments + string literals, lowercase, join continuation lines."""
    logical = []
    pending = ""
    for raw in text.splitlines():
        # strip strings + trailing comment (char walk; respect quotes)
        out = []
        q = None
        i = 0
        while i < len(raw):
            c = raw[i]
            if q:
                if c == q:
                    q = None
                # drop string contents
            else:
                if c in ("'", '"'):
                    q = c
                elif c == "!":
                    break  # comment to end of line
                else:
                    out.append(c)
            i += 1
        line = "".join(out).rstrip().lower()
        # join continuations
        s = (pending + " " + line).strip() if pending else line
        if s.endswith("&"):
            pending = s[:-1].rstrip()
            # a leading '&' on the next physical line is a continuation marker
            continue
        pending = ""
        # drop a leading continuation '&' if present
        s = re.sub(r"^&\s*", "", s)
        logical.append(s)
    if pending:
        logical.append(pending)
    return logical


def count_constructs(lines):
    o = dict(module=0, subroutine=0, function=0, do=0, ifthen=0, type=0)
    c = dict(module=0, subroutine=0, function=0, do=0, ifthen=0, type=0)
    stmts = []
    for s in lines:
        stmts.extend(s.split(";"))      # statement-level (one-liners balance)
    for raw in stmts:
        st = raw.strip()
        if not st:
            continue
        # --- ends (check first; 'end x' must not also count as an open) ------
        m = re.match(r"^end\s*(module|subroutine|function|do|if|type)\b", st)
        if m:
            kind = m.group(1)
            c["ifthen" if kind == "if" else kind] += 1
            continue
        if re.match(r"^endif\b", st):
            c["ifthen"] += 1
            continue
        if re.match(r"^enddo\b", st):
            c["do"] += 1
            continue
        if re.match(r"^end\b", st):  # bare 'end' (program/module terminator) -- ignore
            continue
        # an else / else-if is a continuation of an open if-then: neither open nor close
        if re.match(r"^else\b", st) or re.match(r"^elseif\b", st):
            continue
        # --- opens ----------------------------------------------------------
        # module (definition), not 'module procedure'
        if re.match(r"^module\s+(?!procedure\b)\w+", st):
            o["module"] += 1
        # subroutine (with optional prefixes, incl. CUDA-Fortran attributes(...))
        if re.match(r"^(?:(?:pure|elemental|recursive|impure|module|"
                    r"attributes\s*\([^)]*\))\s+)*"
                    r"subroutine\s+\w+", st):
            o["subroutine"] += 1
        # function definition (with optional type/attr prefixes); 'end function'
        # already consumed above. NOT 'interface'/'use' artefacts (those lack '(').
        # Intrinsic-type prefixes may carry a kind, e.g. 'complex(tp) function f('.
        if re.match(r"^(?:(?:pure|elemental|recursive|impure|module|"
                    r"attributes\s*\([^)]*\)|"
                    r"(?:integer|real|complex|logical|character)\s*(?:\([^)]*\))?|"
                    r"double\s+precision|type\s*\([^)]*\)|class\s*\([^)]*\))\s+)*"
                    r"function\s+\w+\s*\(", st):
            o["function"] += 1
        # do-loop OPEN at LINE START only (matches the original's deliberate gotcha:
        # inner one-liner 'do' after a ';' is NOT counted, mirroring 'end do').
        if re.match(r"^(?:\w+\s*:\s*)?do\b", st):
            o["do"] += 1
        # derived-type DEFINITION: 'type ::', 'type,', or 'type name' -- NOT 'type('
        if re.match(r"^type\b", st):
            rest = st[4:].lstrip()
            if rest[:1] in (":", ","):       # 'type ::' or 'type, ...'
                o["type"] += 1
            elif rest and rest[0].isalpha():  # 'type name'
                o["type"] += 1
            # 'type(' -> a declaration, not a definition: ignore
        # block-if: logical line ends with 'then'
        if re.search(r"\bthen$", st) and re.search(r"\bif\b", st):
            o["ifthen"] += 1
    return o, c


def check_file(path):
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"[FAIL] {path}: {e}")
        return False
    o, c = count_constructs(strip_and_join(text))
    ok = True
    for k in o:
        if o[k] != c[k]:
            ok = False
    tag = "PASS" if ok else "FAIL"
    detail = "  ".join(f"{k}:{o[k]}/{c[k]}" for k in o)
    print(f"[{tag}] {path}  ({detail})")
    return ok


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    allok = True
    for p in argv[1:]:
        allok = check_file(p) and allok
    print("ALL BALANCED" if allok else "STRUCTURAL IMBALANCE")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
