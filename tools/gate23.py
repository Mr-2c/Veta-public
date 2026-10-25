#!/usr/bin/env python3
"""
gate23.py -- case-fold scan (WITHOUT string strip) + device/host audit.
CLAUSON multi-GPU source gate.

Two scans, both ADVISORY (reconcile against the authoritative gate2_clean.py):

  (A) Case-fold collision WITHOUT stripping string literals. This intentionally
      reproduces the known FP where a string like 'ALL TESTS PASSED' collides
      with the all() intrinsic -- gate2_clean.py (string-stripped) is the
      tie-breaker; anything flagged here but clean there is a string FP.

  (B) Device/host heuristic audit, flagging the nvfortran traps:
        - a CUF kernel region (!$cuf kernel ... end do) that dereferences a
          derived-type member  x%member  -> NVFORTRAN-S-0155 risk if the member
          is a device allocatable (pass plain device dummies to helpers instead);
        - a device DUMMY argument declared assumed/deferred-shape (with ':')
          -> device dummies must be explicit-shape, no-intent.
      KNOWN FP: a LOCAL device allocatable (deferred-shape (:,:,:)) is fine --
      only DUMMIES must be explicit-shape -- so locals are not flagged.

Usage:  tools/gate23.py FILE.F90 [FILE2.F90 ...]
Exit:   0 if no device/host findings, 1 otherwise. (The case-fold part is
        printed for reconciliation but does not by itself fail the gate; the
        device/host audit does.)
"""
import re
import sys

IDENT = re.compile(r"[A-Za-z]\w*")


def strip_and_join(text, keep_strings):
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
                elif c == "!" and not raw[i:i+5].lower().startswith("!$cuf"):
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


# ---- (A) case-fold WITHOUT string strip (global) ---------------------------
def casefold_with_strings(text):
    d = {}
    for st in strip_and_join(text, keep_strings=True):
        for tok in IDENT.findall(st):
            d.setdefault(tok.lower(), set()).add(tok)
    return [(low, sorted(s)) for low, s in d.items() if len(s) > 1]


# ---- (B) device/host audit -------------------------------------------------
def header_dummies(st):
    """Return the set of dummy-arg names from a subroutine/function header."""
    low = st.lower()
    if not (re.search(r"\bsubroutine\s+\w+", low) or
            re.search(r"\bfunction\s+\w+\s*\(", low)):
        return None
    m = re.search(r"\((.*?)\)", st)
    if not m:
        return set()
    return {a.strip().lower() for a in m.group(1).split(",") if a.strip()}


def audit_device_host(text):
    lines = strip_and_join(text, keep_strings=False)
    findings = []
    dummies = set()
    in_kernel = False
    kdepth = 0
    for st in lines:
        low = st.lower().strip()
        if not low:
            continue
        # track subprogram scope -> dummy set
        h = header_dummies(st)
        if h is not None:
            dummies = h
        if re.match(r"^end\s*(subroutine|function)\b", low):
            dummies = set()

        # (B1) CUF kernel region member-access
        if "!$cuf" in low and "kernel" in low:
            in_kernel = True
            kdepth = 0
            continue
        if in_kernel:
            if re.match(r"^do\b", low) or re.search(r":\s*do\b", low):
                kdepth += 1
            if re.match(r"^end\s*do\b", low) or re.match(r"^enddo\b", low):
                kdepth -= 1
                if kdepth <= 0:
                    in_kernel = False
            if "%" in st:
                for mm in re.finditer(r"(\w+)\s*%\s*(\w+)", st):
                    findings.append(("cuf-kernel derived-type member access "
                                     f"'{mm.group(1)}%{mm.group(2)}' (0155 risk)", st.strip()))

        # (B2) device dummy must be explicit-shape. A dimension is assumed/
        # deferred-shape ONLY if it is a BARE ':' (no bounds). 'sd(0:n-2)' is
        # explicit (the ':' is a lower:upper bound separator) -> NOT flagged.
        if re.search(r"\bdevice\b", low) and "::" in st and "allocatable" not in low:
            decl = st.split("::", 1)[1]
            for ent in split_top_level(decl):
                nm = re.match(r"\s*(\w+)\s*(?:\((.*)\))?\s*$", ent.strip())
                if not nm:
                    continue
                name = (nm.group(1) or "").lower()
                shape = nm.group(2) or ""
                dims = split_top_level(shape) if shape else []
                if name in dummies and any(dd.strip() == ":" for dd in dims):
                    findings.append((f"device dummy '{name}' is assumed/deferred-shape "
                                     f"(must be explicit-shape)", st.strip()))
    return findings


def split_top_level(s):
    """Split on commas that are NOT inside parentheses."""
    parts, depth, cur = [], 0, []
    for c in s:
        if c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur))
    return parts


def check_file(path):
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"[FAIL] {path}: {e}")
        return False
    cf = casefold_with_strings(text)
    dh = audit_device_host(text)
    if cf:
        print(f"[note] {path}: case-fold-with-strings groups (reconcile vs gate2_clean):")
        for low, spell in cf:
            print(f"        {{{', '.join(spell)}}}")
    if dh:
        print(f"[FAIL] {path}: {len(dh)} device/host finding(s):")
        for msg, src in dh:
            print(f"        {msg}\n            @ {src}")
        return False
    print(f"[PASS] {path}  (device/host audit clean)")
    return True


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    allok = True
    for p in argv[1:]:
        allok = check_file(p) and allok
    print("DEVICE/HOST CLEAN" if allok else "DEVICE/HOST FINDINGS")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
