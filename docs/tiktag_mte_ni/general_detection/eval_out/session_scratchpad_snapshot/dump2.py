import sys, glob
REPO="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, REPO)
vd=sys.argv[1]
from src.aarch64.aarch64_executor_input_encoder import deserialize
for sus in (1,9,17,25):
    inp=deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0],"rb").read()).input_
    g=inp['gpr'][0]
    print(f"input {sus}: gpr[0..7] =", [hex(int(x)) for x in g])
inp=deserialize(open(glob.glob(f"{vd}/input_0001.*")[0],"rb").read()).input_
print("\n-- input1 main[0:6]:", [hex(int(x)) for x in inp['main'][0][0:6]])
print("-- input1 faulty[0:6]:", [hex(int(x)) for x in inp['faulty'][0][0:6]])
# granule g -> which u64? granule=16 bytes=2 u64. main granule m -> main[2m],main[2m+1]
def gran(inp, idx):
    # idx over main(0..255) then faulty(256..511)
    if idx<256:
        reg='main'; j=idx
    else:
        reg='faulty'; j=idx-256
    a=int(inp[reg][0][2*j]); b=int(inp[reg][0][2*j+1])
    return a,b
for idx in (0,255,287,305,286,288,304,306,189,190):
    a,b=gran(inp,idx)
    print(f"-- input1 granule {idx}: lo=0x{a:016x} hi=0x{b:016x}")
