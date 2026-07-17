import os, sys
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.config import CONF
CONF.load('/home/gal_k_1_1998/sca-fuzzer/configs/spectre_v1_fr_remote.yml')
from src import factory
from src.aarch64.aarch64_executor_input_encoder import ExecutorInput

VD = '/home/gal_k_1_1998/sca-fuzzer/violations_fr/violation-260716-181940'
D  = os.path.dirname(os.path.abspath(__file__))
ig = factory.get_input_generator(CONF.input_gen_seed)
base = ig.load([os.path.join(VD, 'input_0009.reif')])[0]

VALUES = [0x1000, 0x1040, 0x1080, 0x10c0, 0x1100, 0x1001, 0x1020]
for V in VALUES:
    inp = base.copy()
    inp[0]['gpr'][1] = V          # x1 = V; everything else identical to #9
    ExecutorInput(input_=inp).save(f'{D}/in_{V:#06x}.reif')
    back = ig.load([f'{D}/in_{V:#06x}.reif'])[0]
    print(f"  wrote in_{V:#06x}.reif  x1 readback={int(back[0]['gpr'][1]):#x}  set={(V//64)%64}")
