import sys, glob
REPO="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, REPO)
vd=sys.argv[1]; sus=int(sys.argv[2])
from src.aarch64.aarch64_executor_input_encoder import deserialize
blob=open(glob.glob(f"{vd}/input_{sus:04d}.*")[0],"rb").read()
ei=deserialize(blob)
inp=ei.input_
print("ExecutorInput type:", type(ei).__name__)
print("Input type:", type(inp).__name__)
print("Input attrs:", [a for a in dir(inp) if not a.startswith('__')])
# Revizor Input is a numpy structured array subclass with named regions
try:
    print("dtype names:", inp.dtype.names)
except Exception as e:
    print("no dtype:", e)
# try to show registers region
for region in ("main","faulty","gpr","registers","reg"):
    try:
        r=inp[region]
        print(f"region {region}: shape={getattr(r,'shape',None)} dtype={getattr(r,'dtype',None)}")
    except Exception:
        pass
