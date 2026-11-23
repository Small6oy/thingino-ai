#!/usr/bin/env python3
"""Run libjzdl.m.so's own model loader on a .bin under Unicorn (no camera).

  jzdl_loadcheck.py libjzdl.m.so MODEL.bin [--trace trace.json] [-v]

Emulates jzdl::net_create() and Net::load_model(path, false), stubbing libc.
Reports load_model's return value and how many bytes it consumed; exit 0 only
if it returned 0 and read the whole file. --trace writes every fread as
[layer_ctx, file_offset, size, count, return_addr, dest].

Only loading is emulated. The forward pass uses MXU1 (XBurst SIMD), which
Unicorn does not implement.

Needs: pip install unicorn capstone pyelftools  (run with python -I).
Addresses below are for libjzdl 5.4.0 "simd" (MD5 in jzdl-format.md).
"""
import sys, struct, json, math, subprocess
import capstone
from unicorn import Uc, UcError, UC_ARCH_MIPS, UC_MODE_MIPS32, UC_MODE_LITTLE_ENDIAN, UC_HOOK_CODE
from unicorn.mips_const import *
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

NET_CREATE = 0x9d74          # jzdl::net_create()
LAYER_TABLE = 0x90b24        # layer creator function pointers, 86 entries
LAYER_TABLE_N = 86
DEFAULT_METHOD = 0x7670      # Layer::load_param/load_model default (returns 0)


class Lib:
    def __init__(self, path):
        self.raw = open(path, 'rb').read()
        self.elf = ELFFile(open(path, 'rb'))
        dyn = {t.entry.d_tag: t.entry.d_val for t in self.elf.get_section_by_name('.dynamic').iter_tags()}
        self.got, self.lno, self.gsym = dyn['DT_PLTGOT'], dyn['DT_MIPS_LOCAL_GOTNO'], dyn['DT_MIPS_GOTSYM']
        self.dsyms = list(self.elf.get_section_by_name('.dynsym').iter_symbols())
        self.md = capstone.Cs(capstone.CS_ARCH_MIPS, capstone.CS_MODE_MIPS32 + capstone.CS_MODE_LITTLE_ENDIAN)

    def a2o(self, a):
        for s in self.elf.iter_segments():
            if s['p_type'] == 'PT_LOAD' and s['p_vaddr'] <= a < s['p_vaddr'] + s['p_filesz']:
                return a - s['p_vaddr'] + s['p_offset']

    def u32(self, a):
        return struct.unpack_from('<I', self.raw, self.a2o(a))[0]

    def _insns(self, a, n):
        o = self.a2o(a)
        return list(self.md.disasm(self.raw[o:o + n], a))

    def _vtables(self, fn):
        """vtable addresses a function materialises from GOT loads (+ addiu)."""
        gp = self.got + 0x7ff0
        regs, out = {}, []
        for i in self._insns(fn, 0x300):
            ops = [x.strip() for x in i.op_str.split(',')]
            if i.mnemonic == 'lw' and len(ops) == 2 and ops[1].endswith('($gp)'):
                regs[ops[0]] = self.u32(gp + int(ops[1][:-5], 0))
            elif i.mnemonic == 'addiu' and len(ops) == 3 and ops[1] in regs:
                v = regs[ops[1]] + int(ops[2], 0)
                regs[ops[0]] = v
                if 0x90000 <= v < 0x92880:
                    out.append(v)
            if i.mnemonic == 'jr' and ops[0] == '$ra':
                break
        return out

    def layer_methods(self):
        """{type: (load_param, load_model)} from the creator table + vtables."""
        res = {}
        for t in range(LAYER_TABLE_N):
            c = self.u32(LAYER_TABLE + 4 * t)
            if not c:
                continue
            ctor = next((int(i.op_str, 16) for i in self._insns(c, 0x60) if i.mnemonic == 'bal'), None)
            vts = (self._vtables(ctor) if ctor else []) or self._vtables(c)
            if vts:
                vt = vts[-1]
                res[t] = (self.u32(vt + 8), self.u32(vt + 12))
        return res


def demangle(n):
    return subprocess.run(['c++filt', n], capture_output=True, text=True).stdout.strip()


def run(libpath, model_bytes, verbose=False):
    lib = Lib(libpath)
    mu = Uc(UC_ARCH_MIPS, UC_MODE_MIPS32 + UC_MODE_LITTLE_ENDIAN)
    mu.mem_map(0, 0xa0000)
    for s in lib.elf.iter_segments():
        if s['p_type'] == 'PT_LOAD':
            mu.mem_write(s['p_vaddr'], lib.raw[s['p_offset']:s['p_offset'] + s['p_filesz']])
    STUB, HEAP, STK, RET = 0x100000, 0x1000000, 0x20000000, 0x0ffffff0
    mu.mem_map(STUB, 0x10000)
    mu.mem_map(HEAP, 0x8000000)
    mu.mem_map(STK - 0x100000, 0x100000)
    mu.mem_map(RET & ~0xffff, 0x10000)
    heap = [HEAP]
    stubs, names = {}, {}

    def symaddr(s):
        if s['st_shndx'] != 'SHN_UNDEF' and s['st_value']:
            return s['st_value']
        if s.name not in stubs:
            a = STUB + 4 * len(stubs)
            stubs[s.name] = a
            names[a] = demangle(s.name)
        return stubs[s.name]

    for i, s in enumerate(lib.dsyms[lib.gsym:]):
        mu.mem_write(lib.got + 4 * (lib.lno + i), struct.pack('<I', symaddr(s)))
    for sec in lib.elf.iter_sections():
        if isinstance(sec, RelocationSection):
            for r in sec.iter_relocations():
                if r['r_info_type'] == 3 and r['r_info_sym']:      # R_MIPS_REL32 vs symbol
                    o = r['r_offset']
                    old = struct.unpack('<I', mu.mem_read(o, 4))[0]
                    mu.mem_write(o, struct.pack('<I', (old + symaddr(lib.dsyms[r['r_info_sym']])) & 0xffffffff))

    R = mu.reg_read
    def W(r, v): mu.reg_write(r, v & 0xffffffff)
    fpos, trace, ctx = [0], [], ['header']

    def alloc(n):
        a = (heap[0] + 15) & ~15
        heap[0] = a + max(n, 1)
        return a

    def cstr(a):
        b = bytearray()
        while True:
            c = mu.mem_read(a, 1)[0]
            if not c:
                return b.decode('latin1')
            b.append(c); a += 1

    def ret(v=0):
        W(UC_MIPS_REG_V0, v); W(UC_MIPS_REG_PC, R(UC_MIPS_REG_RA))

    def getd(r1, r2):
        return struct.unpack('<d', struct.pack('<II', R(r1) & 0xffffffff, R(r2) & 0xffffffff))[0]

    def setd(v):
        lo, hi = struct.unpack('<II', struct.pack('<d', v))
        W(UC_MIPS_REG_F0, lo); W(UC_MIPS_REG_F1, hi)

    def stub(uc, addr, size, ud):
        n = names.get(addr, '?')
        a0, a1, a2 = R(UC_MIPS_REG_A0), R(UC_MIPS_REG_A1), R(UC_MIPS_REG_A2)
        if n == 'fopen':
            fpos[0] = 0; return ret(0x4000)
        if n == 'fread':
            data = model_bytes[fpos[0]:fpos[0] + a1 * a2]
            trace.append([ctx[0], fpos[0], a1, a2, R(UC_MIPS_REG_RA), a0])
            mu.mem_write(a0, data); fpos[0] += len(data)
            return ret(len(data) // a1 if a1 else 0)
        if n in ('fclose', '__cxa_atexit', 'printf', 'puts', 'fwrite', 'std::ios_base::Init::Init()'):
            return ret(0)
        if n == 'fprintf':
            if verbose:
                print('fprintf:', cstr(a1).rstrip(), file=sys.stderr)
            return ret(0)
        if n in ('malloc', 'operator new(unsigned int)', 'operator new[](unsigned int)'):
            return ret(alloc(a0))
        if n == 'calloc':
            a = alloc(a0 * a1); mu.mem_write(a, b'\0' * (a0 * a1)); return ret(a)
        if n in ('free', 'operator delete(void*)', 'operator delete[](void*)'):
            return ret(0)
        if n == 'memset':
            mu.mem_write(a0, bytes([a1 & 0xff]) * a2); return ret(a0)
        if n in ('memcpy', 'memmove'):
            mu.mem_write(a0, bytes(mu.mem_read(a1, a2))); return ret(a0)
        if n == 'strlen':
            return ret(len(cstr(a0)))
        if n == 'atoi':                       # net_create parses its version string
            s = cstr(a0).strip()
            return ret(int(s) if s.lstrip('-').isdigit() else 0)
        if n in ('ceil', 'floor', 'exp'):
            setd(getattr(math, n)(getd(UC_MIPS_REG_F12, UC_MIPS_REG_F13)))
            W(UC_MIPS_REG_PC, R(UC_MIPS_REG_RA)); return
        print('unhandled import %s (ra=%x)' % (n, R(UC_MIPS_REG_RA)), file=sys.stderr)
        uc.emu_stop()

    mu.hook_add(UC_HOOK_CODE, stub, begin=STUB, end=STUB + 0xffff)

    lp, lm = {}, {}
    for t, (p, m) in lib.layer_methods().items():
        if p != DEFAULT_METHOD: lp[p] = t
        if m != DEFAULT_METHOD: lm[m] = t
    li = [-1]

    def mark(uc, addr, size, ud):
        if addr in lp:
            li[0] += 1; ctx[0] = 'L%d:t%d:param' % (li[0], lp[addr])
        elif addr in lm:
            ctx[0] = 't%d:model' % lm[addr]
    for a in set(lp) | set(lm):
        mu.hook_add(UC_HOOK_CODE, mark, begin=a, end=a)

    def call(fn, args):
        for r, v in zip((UC_MIPS_REG_A0, UC_MIPS_REG_A1, UC_MIPS_REG_A2, UC_MIPS_REG_A3), args):
            W(r, v)
        W(UC_MIPS_REG_SP, STK - 0x1000); W(UC_MIPS_REG_T9, fn); W(UC_MIPS_REG_RA, RET)
        mu.emu_start(fn, RET)
        return R(UC_MIPS_REG_V0)

    rv = None
    try:
        net = call(NET_CREATE, [])
        vt = struct.unpack('<I', mu.mem_read(net, 4))[0]
        load_model = struct.unpack('<I', mu.mem_read(vt + 8, 4))[0]
        p = alloc(16); mu.mem_write(p, b'model\0')
        rv = call(load_model, [net, p, 0])
        if rv & 0x80000000:
            rv -= 1 << 32
    except UcError as ex:
        print('emulation error: %s at pc=%x' % (ex, R(UC_MIPS_REG_PC)), file=sys.stderr)
    return rv, fpos[0], trace


def main(argv):
    if len(argv) < 3:
        print(__doc__); return 2
    data = open(argv[2], 'rb').read()
    rv, used, trace = run(argv[1], data, '-v' in argv)
    ok = rv == 0 and used == len(data)
    print('load_model -> %s, consumed %d of %d bytes: %s' % (rv, used, len(data), 'OK' if ok else 'FAIL'))
    if '--trace' in argv:
        json.dump(trace, open(argv[argv.index('--trace') + 1], 'w'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
