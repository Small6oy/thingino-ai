import sys, capstone, subprocess
from elftools.elf.elffile import ELFFile
f=open(sys.argv[1],'rb'); elf=ELFFile(f); data=open(sys.argv[1],'rb').read()
syms={}
for sec in elf.iter_sections():
    if sec.header.sh_type in ('SHT_SYMTAB','SHT_DYNSYM'):
        for s in sec.iter_symbols():
            if s['st_value'] and s.name: syms.setdefault(s['st_value'],s.name)
names=sorted(syms.items())
dem={}
try:
    out=subprocess.run(['c++filt'],input='\n'.join(n for _,n in names),capture_output=True,text=True).stdout.split('\n')
    for (a,n),d in zip(names,out): dem[a]=d
except: dem={a:n for a,n in names}
# GOT resolution for MIPS: local + global entries
dyn=elf.get_section_by_name('.dynamic')
tags={t.entry.d_tag:t.entry.d_val for t in dyn.iter_tags()}
gotaddr=tags.get('DT_PLTGOT'); localno=tags.get('DT_MIPS_LOCAL_GOTNO'); gotsym=tags.get('DT_MIPS_GOTSYM')
dynsym=elf.get_section_by_name('.dynsym'); dsl=list(dynsym.iter_symbols())
got=elf.get_section_by_name('.got')
gp=gotaddr+0x7ff0
def gotname(off):
    a=gp+off; idx=(a-gotaddr)//4
    if idx>=localno:
        s=dsl[gotsym+idx-localno]; return subprocess.run(['c++filt',s.name],capture_output=True,text=True).stdout.strip()
    v=int.from_bytes(data[got['sh_offset']+idx*4:got['sh_offset']+idx*4+4],'little')
    return 'local:%x'%v
def addr2off(a):
    for seg in elf.iter_segments():
        if seg['p_type']=='PT_LOAD' and seg['p_vaddr']<=a<seg['p_vaddr']+seg['p_filesz']:
            return a-seg['p_vaddr']+seg['p_offset']
start=int(sys.argv[2],16); n=int(sys.argv[3],16) if len(sys.argv)>3 else 0x400
md=capstone.Cs(capstone.CS_ARCH_MIPS,capstone.CS_MODE_MIPS32+capstone.CS_MODE_LITTLE_ENDIAN)
o=addr2off(start)
for i in md.disasm(data[o:o+n],start):
    if i.address in dem: print('\n<%s>:'%dem[i.address])
    c=''
    if '($gp)' in i.op_str and i.mnemonic=='lw':
        try:
            off=int(i.op_str.split(',')[1].strip().split('(')[0],0); c='  ; '+gotname(off)
        except Exception as e: c='  ; ?'
    print('%08x: %-8s %s%s'%(i.address,i.mnemonic,i.op_str,c))
