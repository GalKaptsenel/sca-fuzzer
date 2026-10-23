import struct
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
D="/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/CONTESTED/fuzzer_072421"
b=open(f"{D}/enacted_test0001.bin","rb").read()
md=Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
seals={0xf8:"<<SEAL A (x4, gates next LDR)",0x10c:"<<SEAL B1 (x1)",0x114:"<<SEAL B2 (x1)",0x120:"<<SEAL D (x2, gates next LDP)"}
for ins in md.disasm(b,0):
    off=ins.address
    note=seals.get(off,"")
    print(f"  0x{off:03x}: {ins.mnemonic:8s} {ins.op_str:28s} {note}")
