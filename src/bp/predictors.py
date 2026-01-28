from copy import deepcopy
from multiprocessing.util import DEBUG
import sys
import math
import random

import sys
import math

from abc import ABC, abstractmethod
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from typing import List, Optional
#from ..config import CONF

#from ...helpers import PROF
#TODO: move here (because problem of importing)
# import utils
DEBUG_PRINT = False
class StateCounter:
    def __init__(self, bits, init_value):
        self.bits = bits
        self.max_val = 2 ** self.bits - 1
        self.init_value = init_value

        self.state = self.init_value

    def was_taken(self):
        if self.state == self.max_val:
            return
        self.state += 1

    def was_not_taken(self):
        if self.state == 0:
            return
        self.state -= 1

    def get_state(self):
        if self.state >= ((self.max_val + 1) / 2):  # upper side
            return 1
        else:
            return 0

    def get_confidence(self):
        return self.state
        weak_states = [((self.max_val + 1) / 2)-1, ((self.max_val + 1) / 2)]
        if self.state <= weak_states[0]:
            return (weak_states[0]-self.state)
        return (self.state-weak_states[1])

    def __eq__(self, other):
        if isinstance(other, StateCounter):
            return self.state == other.state and self.init_value == other.init_value \
            and self.max_val == other.max_val
        return False

#TODO: for now will store only branch (with no counter)
class IndirectState:
    def __init__(self, target_bits, init_value = 0x0):
        self.bits = target_bits # which pc bits would be saved
        self.init_value = init_value
        self.target = self.init_value

    def update_prediction(self, target_addr):
        self.target = ((1<<self.bits) -1) & target_addr

    def get_state(self):
        if self.target == 0x0: # SPECIAL VALUE MEANS NO PREDICTION
            return None

        return self.target


class PredictorCounter(StateCounter):
    def __init__(self, bits, init_value):
        super().__init__(bits, init_value)

    def get_state(self):
        if self.state >= ((self.max_val + 1) / 2):
            return 1  # Taken
        if self.state < (((self.max_val + 1) / 2)):  # fix bug
            return 0  # Not Taken
        else:  # NOTE: don't think so
            return None  # No Prediction (Weak)

#TODO: update to use pattern of phr branch (from paper)
class ShiftRegister:
    def __init__(self, bits):
        self.max_bits = bits
        self.register = [0 for i in range(bits)]

    def shift_in(self, bit):
        self.register.pop(0)
        self.register.append(bit)

    def get_current_val(self):
        return int("".join(map(str, self.register)), 2)

    def get_current_val_as_binstr(self):
        return str("".join(map(str, self.register)))

class ShiftRegisterLSB:
    def __init__(self, bits):
        self.max_bits = bits
        self.register = [0 for i in range(bits)]

    def shift_in(self, bit):
        self.register = self.register[:-1]
        self.register.insert(bit)

    def get_current_val(self):
        return int("".join(map(str, self.register[::-1])), 2)

    def get_current_val_as_binstr(self):
        return str("".join(map(str, self.register[::-1])))


#returns copy of result
def xor_lists(list1, list2):
    res = []
    if len(list1) > len(list2):
        res = deepcopy(list1)
    else:
        res = deepcopy(list2)

    for i in range(min(len(list1), len(list2))): # 0 it's lsb
        res[i] = list1[i] ^ list2[i]

    return res
#TODO: don't really need inheritance
class PathHistoryRegister(ShiftRegisterLSB):
        def __init__(self, max_bits=388):
            # self.__init__(max_bits)
            self.max_bits = max_bits
            self.register = bitarray(max_bits,endian="little")

        def update(self, branch_addr: int, target_addr: int):
            # shift to left by
            if DEBUG_PRINT:
                print(f"Path Before branch: {self.register}")
            self.register >>= 2 # NOTE: it's right because we are using "little endian array"
            # print(f"After shift: {self.register}")


            # XOR in-place with footprint over the first 16 bits
            self.register[:16] ^= self.footprint(branch_addr, target_addr)
            # print(f"Branch Address: {hex(branch_addr)}, Target Address: {hex(target_addr)}")
            # print("Footprint: ", bin(ba2int(self.footprint(branch_addr, target_addr)))[2:][::-1])
            # print(f"After: {self.register}")
        @staticmethod
        def footprint(branch_addr: int, target_addr: int) -> bitarray:
            FOOTPRINT_SIZE = 16

            #int to list
            b = int2ba(branch_addr & 0xFFFF, length=FOOTPRINT_SIZE, endian='little')
            t = int2ba(target_addr & 0xFFFF, length=FOOTPRINT_SIZE, endian='little')

            # magic number for 16
            branch_vector = (
                b[3:11] +         # bits 3–10 → positions 0–7
                b[0:3] +          # bits 0–2 → positions 8–10
                b[11:16]          # bits 11–15 → positions 11–15
            )
            target_vector = bitarray(FOOTPRINT_SIZE,endian='little')
            target_vector [0:2] = t[0:2]
            target_vector [8:12] = t[2:6]

            return branch_vector ^ target_vector

class BranchPredictor(ABC):
    def __init__(self, init_state_val, pht_size):
        self.init_state_val = init_state_val
        self.pht_size = pht_size


    @abstractmethod
    def predict(self, pc, actual_result):
        pass

    @abstractmethod
    def prediction_method(self, cutpc, actual_branch):
        pass

    def get_method_type(self):
        return type(self).__name__.rstrip()



class CondBranchPredictor(BranchPredictor):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__( init_state_val, pht_size)
        self.num_state_bits = num_state_bits
        offset = 0
        self.pht_numbits = math.frexp(pht_size)[1] - 1
        self.cut_pc = [self.pht_numbits + offset, offset]
        self.pattern_history_table = [PredictorCounter(num_state_bits, init_state_val)
                                      for _ in range(pht_size)]

        #TODO:Get rid of this basic stupid call
        init_basic_vars(self, num_state_bits, init_state_val, pht_size) # NOTE: why not in super.()?

        self.count = 0

    def predict(self, pc, actual_branch):
        cutpc = get_from_bitrange(self.cut_pc, pc)  # NOTE: here it calculates pht entry

        prediction = self.prediction_method(cutpc, actual_branch)
        return prediction

    @abstractmethod
    def prediction_method(self, cutpc, actual_branch):
        pass

    def get_method_type(self):
        return type(self).__name__.rstrip()


#TODO: update it to behave as BTB(collisions and more)
class TAGEBimodalBase(CondBranchPredictor):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__(num_state_bits, init_state_val, pht_size)
        self.pattern_history_table = [StateCounter(num_state_bits, init_state_val)
                                      for i in range(pht_size)]

    def prediction_method(self, cutpc, actual_branch):
        pht_address = cutpc
        prediction = self.pattern_history_table[pht_address].get_state()
        return prediction

    def update(self, pc, actual_branch):
        cutpc = get_from_bitrange(self.cut_pc, pc)
        pht_address = cutpc
        if actual_branch == 1:
            self.pattern_history_table[pht_address].was_taken()
        elif actual_branch == 0:
            self.pattern_history_table[pht_address].was_not_taken()


class TaggedTable:
    def __init__(self, num_state_bits, init_state_val):
        self.index_bits = 10
        num_entries = 2 ** self.index_bits
        self.tag_width = 8

        self.counters = [StateCounter(num_state_bits, init_state_val)
                         for i in range(num_entries)]
        self.tags = [0 for i in range(num_entries)]
        self.useful_bits = [StateCounter(2, 0) for i in range(num_entries)]

    def predict(self, index, actual_branch):
        prediction = self.counters[index].get_state()
        return prediction

    def update(self, index, actual_branch):
        if actual_branch == 1:
            self.counters[index].was_taken()
        elif actual_branch == 0:
            self.counters[index].was_not_taken()

    def get_tag_at(self, index):
        return self.tags[index]


def print_stats(predictor):
    total = predictor.no_predictions + predictor.good_predictions + predictor.mispredictions
    print("\n\n\n\t\t---Sim Result---")
    print("Type\t\t", "Counter Bits\t", "Counter init\t", "PHT entries")
    print(predictor.get_method_type(), "\t", predictor.num_state_bits, "\t\t",
          predictor.init_state_val, "\t\t", predictor.pht_size, "\n")
    print("Mispredictions:\t\t", predictor.mispredictions)
    print("No Predictions:\t\t", predictor.no_predictions)
    print("Hit Predictions:\t", predictor.good_predictions)
    print("Total:\t\t\t", total)
    # print("Hit rate:\t\t", '{0:.04f}'.format(self.good_predictions / (total - self.no_predictions) * 100), "%")
    print("Hit rate:\t\t", '{0:.04f}'.format(predictor.good_predictions / (total) * 100), "%")
    print("MP/KI:\t\t\t",
          '{0:.04f}'.format((predictor.mispredictions + predictor.no_predictions) / total * 1000),
          '\n')

    # disp_big_list(predictor.T[1].tags)
    # disp_big_list(predictor.T[2].tags)
    # disp_big_list(predictor.T[3].tags)
    # disp_big_list(predictor.T[4].tags)


def init_basic_vars(predictor, num_state_bits, init_state_val, pht_size):
    predictor.num_state_bits = num_state_bits
    predictor.init_state_val = init_state_val
    predictor.pht_size = pht_size
    # predictor.mispredictions = 0
    # predictor.good_predictions = 0
    # predictor.no_predictions = 0


def norm_branch(branch):
    return 1 if branch.rstrip() == 'T' else 0

#NOTE: what a stupid way to do & (who wrote this?)
def get_from_bitrange(bit_range, dec_val):
    left_bit, right_bit = bit_range
    binary_string = "{0:064b}".format(int(dec_val))
    left_bit = len(binary_string) - left_bit
    right_bit = len(binary_string) - right_bit
    cut_string = binary_string[left_bit:right_bit]
    return 0 if left_bit == right_bit else int(cut_string, 2)


def binstr_get_from_bitrange(bit_range, binary_string):
    left_bit, right_bit = bit_range
    left_bit = len(binary_string) - left_bit
    right_bit = len(binary_string) - right_bit
    cut_string = binary_string[left_bit:right_bit]
    return 0 if left_bit == right_bit else int(cut_string, 2)


def disp_big_list(lst, rows=50):
    table_list = [[] for _ in range(rows)]

    for index, item in enumerate(lst):
        row_index = index % rows
        table_list[row_index].append("%6d" % item)

    table_str = "\n".join(["\t".join(i) for i in table_list])

    print(table_str)

class OneLevel(CondBranchPredictor):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__(num_state_bits, init_state_val, pht_size)

    def prediction_method(self, cutpc, actual_branch):
        pht_address = cutpc
        prediction = self.pattern_history_table[pht_address].get_state(),self.pattern_history_table[pht_address].state

        if actual_branch == 1:
            self.pattern_history_table[pht_address].was_taken()
        elif actual_branch == 0:
            self.pattern_history_table[pht_address].was_not_taken()

        return prediction
#NOTE: global history register - don't use at all the branch pc
class TwoLevelGlobal(CondBranchPredictor):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__(num_state_bits, init_state_val, pht_size)

        self.g_hist_reg_width = self.pht_numbits
        self.global_branch_history = ShiftRegister(self.g_hist_reg_width)

    def prediction_method(self, cutpc, actual_branch):
        pht_address = self.addressing_method(cutpc)
        prediction = self.pattern_history_table[pht_address].get_state()

        self.global_branch_history.shift_in(actual_branch)
        if actual_branch == 1:
            self.pattern_history_table[pht_address].was_taken()
        elif actual_branch == 0:
            self.pattern_history_table[pht_address].was_not_taken()

        return prediction

    def addressing_method(self, cutpc):
        return self.global_branch_history.get_current_val()

class GShare(TwoLevelGlobal):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__(num_state_bits, init_state_val, pht_size)

    def addressing_method(self, cutpc):
        pht_addr = cutpc ^ self.global_branch_history.get_current_val()
        # DEBUG print
        # with open("log.txt","a") as f:
        #     f.write("cutpc: %d, ghr: %d, pht_addr: %d\n" % (cutpc, self.global_branch_history.get_current_val(), pht_addr))
        return pht_addr
    def print_debug_stats(self):
        print("\n---Debug---")
        print("Bits in history register:\t\t", self.g_hist_reg_width)
        print("Current values in global history reg:\t", self.global_branch_history.register)
        print("Value of global history reg:\t\t", self.global_branch_history.get_current_val())

#NOTE: local history register
class TwoLevelLocal(CondBranchPredictor):
    def __init__(self, num_state_bits, init_state_val, pht_size):
        super().__init__(num_state_bits, init_state_val, pht_size)
        self.g_hist_reg_width = self.pht_numbits

        self.local_hist_reg_table_size =  128

        self.reg_table_numbits = math.frexp(self.local_hist_reg_table_size)[1] - 1
        self.cut_pc = [32, 32 - self.reg_table_numbits]

        self.local_hist_reg_table = [ShiftRegister(self.g_hist_reg_width)
                for _ in range(self.local_hist_reg_table_size)]

    def prediction_method(self, cutpc, actual_branch):
        pht_address = self.local_hist_reg_table[cutpc].get_current_val()
        prediction = self.pattern_history_table[pht_address].get_state()

        self.local_hist_reg_table[cutpc].shift_in(actual_branch)
        if actual_branch == 1:
            self.pattern_history_table[pht_address].was_taken()
        elif actual_branch == 0:
            self.pattern_history_table[pht_address].was_not_taken()

        return prediction

class TournamentPredictor:
    def __init__(self, num_state_bits, init_state_val, pht_size):
        offset = 0
        self.pht_numbits = math.frexp(pht_size)[1] - 1
        self.cut_pc = [self.pht_numbits + offset, offset]

        gshare_predictor = GShare(num_state_bits, init_state_val, pht_size)
        one_level_predictor = OneLevel(num_state_bits, init_state_val, pht_size)
        self.predictors = [gshare_predictor, one_level_predictor]
        self.meta_predictor = [StateCounter(num_state_bits, init_state_val)
                for i in range(pht_size)]

        init_basic_vars(self, num_state_bits, init_state_val, pht_size)

    def predict(self, pc, actual_branch):
        cutpc = get_from_bitrange(self.cut_pc, pc)
        choosen_predictor = self.meta_predictor[cutpc].get_state()
        predictions = [self.predictors[0].predict(pc, actual_branch), self.predictors[1].predict(pc, actual_branch)]
        chosen_prediction = predictions[choosen_predictor]

        if chosen_prediction == actual_branch:
            self.good_predictions += 1
        elif chosen_prediction is not None:
            self.mispredictions += 1
        elif chosen_prediction is None:
            self.no_predictions += 1

        if (predictions[0] == predictions[1]):
            pass
        elif (predictions[0] == actual_branch):
            self.meta_predictor[cutpc].was_not_taken()
        elif (predictions[1] == actual_branch):
            self.meta_predictor[cutpc].was_taken()

    def get_method_type(self):
        return type(self).__name__.rstrip()

# ==========================================================================================
#
#                                 BTB (joint)
#
# ==========================================================================================

class BTBEntry:
    def __init__(self, tag, target, type):
        self.tag = tag
        self.target = target
        self.type = type


class BTB:
    def __init__(self, sets_number = pow(2,10), ways_number = 12):
        self.sets_number = sets_number
        self.ways_number = ways_number
        self.entries : List[List[BTBEntry]] = [[] for _ in range(sets_number)]

        # DEBUG
        self.occupied_entries = {}

    #Indirector Table 1.
    def calc_index(self, pc:bitarray) -> int:
        return ba2int(pc[5:15])

    #Indirector Table 1.
    def calc_tag(self, pc:bitarray) -> int:
        return ba2int(pc[15:24] + pc[0:5])

    def search(self, pc: bitarray):
        index = self.calc_index(pc)
        tag = self.calc_tag(pc)

        ways = self.entries[index]
        for entry in ways:
            if entry.tag == tag:
                return entry
        return None

    def update(self, pc: bitarray, type: str, target: bitarray = None):
        index = self.calc_index(pc)
        tag = self.calc_tag(pc)

        ways = self.entries[index]
        for i, entry in enumerate(ways):
            if entry.tag == tag:
                temp_entry = self.entries[index].pop(i)
                temp_entry.target = target
                # self.entries[index].insert(i)   # i or 0?
                self.entries[index].insert(0, temp_entry) ### NOTE: fixed bug

    def allocate(self, pc: bitarray, type: str, target: bitarray = None):
        index = self.calc_index(pc)
        tag = self.calc_tag(pc)

        entry = BTBEntry(tag, target, type)
        self.entries[index] = self.entries[index][:self.ways_number-1] + [entry]

        # DEBUG
        self.occupied_entries[(index, tag)]=entry

    def predict(self, pc: bitarray, phr : bitarray): # TODO: don't support type for now
        prediction = None
        index = self.calc_index(pc,phr)
        tag = self.calc_tag(pc, phr)
        for i in range(len(self.table[index])):
            if self.table[index][i].get_tag() == tag:
                prediction = self.table[index][i].get_state()
                # NOTE: update LRU
                self.table[index].insert(0, self.table[index].pop(i))
                break
        return prediction # NOTE: NOT necessarily the target address (but could be only partial)

    # DEBUG
    def print_btb(self, base_address: int):
        print("\n=== BTB State ===")
        print(f"{'Index':>6} | {'Tag':>10} | {'Target (hex)':>18}")
        print("-" * 40)
        if self.occupied_entries != {}:
            for (index, tag), entry in self.occupied_entries.items():
                target_val = entry.target
                print(f"{index:>6} | {tag:>10} | {hex(target_val-base_address):>18}")
        else:
            print("No entries occupied yet.\n")


# ==========================================================================================
#
#                                 Condidtional Branch Predictor (new)
#
# ==========================================================================================


class ComponentEntry:
    def __init__(self, bits: int, init_value: int, target: int = None, tag: int = None):
        self.counter = StateCounter(bits=bits, init_value=init_value)
        self.tag = tag

        # Debug
        self.target = target

    def update(self, actual_branch: int, target: int = None):
        if actual_branch == 1:
            self.counter.was_taken()
        else:
            self.counter.was_not_taken()
        self.target = target

class ConditionalBaseComponent:
    def __init__(self, entries_num = 4096):
        self.cut_pc = [0, math.frexp(entries_num)[1] - 1]
        self.entries : List[ComponentEntry] = [None for _ in range(entries_num)]

        # DEBUG
        self.occupied_entries = {}

    def calc_index(self, pc:bitarray) -> int:
        return ba2int(pc[self.cut_pc[0]:self.cut_pc[1]])

    def predict(self, pc: bitarray):
        index = self.calc_index(pc)
#        if CONF.dbg_predictor:
#            print(f"PC: 0x{int(pc.to01()[::-1], 2):x}")
#            print(f"INDEX: 0x{index:x}")
        entry = self.entries[index]
        if entry:
            state, confidence = entry.counter.get_state(), entry.counter.get_confidence()
#            if CONF.dbg_predictor:
#                print(f'ENTRY FOUND')
#                print(f'STATE: {entry.counter.state} OUTOF: {entry.counter.max_val} => {"TAKEN" if state == 1 else "NOT TAKEN"}')
#                print(f'CONFIDENCE: {confidence}')

            return state, confidence

#        if CONF.dbg_predictor:
#            print("NO ENTRY FOUND IN BASE PRED- SHOULDN'T BE REACHED?")

        return None # Shouldn't be possible because we only query base if allocated in it before (after BTB miss)

    def update(self, pc: bitarray, actual_branch: int, target: int):
        index = self.calc_index(pc)
        if self.entries[index]:
            self.entries[index].update(actual_branch, target)

    def allocate(self, pc: bitarray, actual_branch: int, target: int):
        index = self.calc_index(pc)
        self.entries[index] = ComponentEntry(bits=2, init_value=2 + actual_branch, target=target)
        # DEBUG
        self.occupied_entries[index] = self.entries[index]

def get_bit(x, i):
    if i == -1:
        return 0
    return x[i]

def build_bitarray_from_indexes(x, indices):
    arr = bitarray(len(indices), endian="little")
    for i,bit_idx in enumerate(indices):
        if bit_idx == -1:
            bit = 0
        else:
            bit = get_bit(x, bit_idx)
        arr[i] = bit
    return arr

PHR_MAX_SIZE_COND = 194 * 2 # for odds
TAG_LENGTH_COND = 11
INDEX_LENGTH_COND = 9

#defined = False
class ConditionalTaggedComponent:
    def __init__(self, history_length, sets_number = pow(2,9), ways_number = 4):
        self.history_length = history_length
        self.sets_number = sets_number
        self.ways_number = ways_number
        self.index_length = int(math.log(sets_number, 2))
        self.entries : List[List[ComponentEntry]] = [[] for _ in range(sets_number)]
#        global defined
#        if PROF.enable and not defined:
#            ConditionalTaggedComponent.calc_index = PROF.timed("ConditionalTaggedComponent.calc_index")(ConditionalTaggedComponent.calc_index)
#            ConditionalTaggedComponent.calc_tag = PROF.timed("ConditionalTaggedComponent.calc_tag")(ConditionalTaggedComponent.calc_tag)
#            ConditionalTaggedComponent.predict = PROF.timed("ConditionalTaggedComponent.predict")(ConditionalTaggedComponent.predict)
#            defined = True
#        else:
#            defined = True
        # DEBUG
        self.occupied_entries = {}

    def calc_index(self, pc: bitarray, phr: bitarray, phr_slice_length = PHR_MAX_SIZE_COND) -> int:
        #initialize the even & odd bits to 0 in correct size (exclude 9th bith, it's supposed to be pc[5])
        # even_bits = bitarray(INDEX_LENGTH_COND - 1, endian="little")
        # odd_bits = bitarray(INDEX_LENGTH_COND - 1, endian="little")
        index = bitarray(INDEX_LENGTH_COND, endian="little")
        #index[:] = 0 # NOTE: removed not as not needed
        index_folding_length = 8

        last_phr_bit_ind = phr_slice_length - 1
        if phr_slice_length <= 68: # PHT 0
            i_range = range(6, 21, 2)
            j_range = range(1, 16, 2)
            for k, (i, j) in enumerate(zip(i_range, j_range)):
                index[k] = phr[i] ^ phr[j]
            index[8] = pc[5] # 9th bit is always pc[5]
            return ba2int(index) ####### NOTE: BUG FIX


        elif phr_slice_length < PHR_MAX_SIZE_COND:
            # i_range = range(1,4)
            even_range = [1,3]
            # j_range = range(0,4)
            odd_range = [0,3]
        else:
            even_range = [1, 11]
            odd_range = [0, 11]

        for iteration, i in enumerate(range(10, 16*even_range[-1]+9, 2)):
            bit_index = iteration % index_folding_length
            index[bit_index] ^= get_bit(phr, i)


        for iteration, i in enumerate(range(3, 16*odd_range[-1]+2, 2)):
            bit_index = iteration % index_folding_length
            index[bit_index] ^= get_bit(phr, i)


        # for i in i_range:
        #     even_phr_slice = range(16*i-6, 16*i+9, 2)
        #     for j in j_range:
        #         odd__phr_slice = range(16*j-13, 16*j+2,2)
        #         for k, (e,d) in enumerate(zip(even_phr_slice, odd__phr_slice)):
        #             index[k] ^= get_bit(phr, e) ^ get_bit(phr, d)

        index[8] = pc[5]
        return ba2int(index)

    def calc_tag(self, pc: bitarray, phr: bitarray, phr_slice_length:int = PHR_MAX_SIZE_COND) -> int:
        """
        Compute the 11-bit IBP tag from a 388-bit PHR and 16-bit PC using folding logic.
        """
        tag = 0
        even_bits = bitarray(TAG_LENGTH,endian="little")
        odd_bits = bitarray(TAG_LENGTH,endian="little")

        #first one - 10 bits
        for iteration, i in enumerate(range(16,35, 2)):
            offset = TAG_LENGTH - 10
            bit_index = (iteration + offset) % TAG_LENGTH
            even_bits[bit_index] = get_bit(phr, i)
            odd_bits[bit_index] = get_bit(phr, i + 1)

        for iteration, i in enumerate(range(36, phr_slice_length-1, 2)):
            bit_index = iteration % TAG_LENGTH
            even_bits[bit_index] ^= get_bit(phr, i)
            odd_bits[bit_index] ^= get_bit(phr, i + 1)

        odd_bits = odd_bits[5:] + odd_bits[:5] # do a rotation

        lower_phr_bits = bitarray(TAG_LENGTH,endian="little")

        first = [4,6,8,0,2,12,14,-1,-1,10,-1]
        second = [15,-1,-1,11,13,3,5,7,9,1,-1]
        lower_phr_bits = build_bitarray_from_indexes(phr, first) ^ \
            build_bitarray_from_indexes(phr, second)

        pc_bits_indices = [10,12,14,6,8,9,11,13,15,7,-1]
        pc_bits = build_bitarray_from_indexes(pc, pc_bits_indices)

        tag = pc_bits ^ lower_phr_bits ^ even_bits ^ odd_bits
        if DEBUG_PRINT:
            print(f"Tag: {tag}")
        return ba2int(tag)

    def predict(self, pc: bitarray, phr: bitarray):
        index = self.calc_index(pc, phr, self.history_length)
        tag = self.calc_tag(pc, phr, self.history_length)
#        if CONF.dbg_predictor:
#            print(f"PC: 0x{int(pc.to01()[::-1], 2):x}")
#            print(f"INDEX: 0x{index:x}")
#            print(f"TAG: 0x{tag:x}")
#            print(f"PHR(relevant length={self.history_length}): {int(phr.to01(), 2):0{PHR_MAX_SIZE}b}")
        ways = self.entries[index]

        for i, entry in enumerate(ways):
            if entry.tag == tag:
                state, confidence = entry.counter.get_state(), entry.counter.get_confidence()
#                if CONF.dbg_predictor:
#                    print(f'TAG MATCH')
#                    print(f'STATE: {entry.counter.state} OUTOF: {entry.counter.max_val} => {"TAKEN" if state == 1 else "NOT TAKEN"}')
#                    print(f'CONFIDENCE: {confidence}')
                return state, confidence

#        if CONF.dbg_predictor:
#            print("NO TAG MATCH")
        return None

    def update(self, pc: bitarray, phr: bitarray, actual_branch: int, target: int):
        index = self.calc_index(pc, phr, self.history_length)
        tag = self.calc_tag(pc, phr, self.history_length)

        ways = self.entries[index]

        # TODO: remove the usefulness
        for i, entry in enumerate(ways):
            if entry.tag == tag:
                temp = self.entries[index].pop(0)
                temp.update(actual_branch, target)
                self.entries[index].insert(0, temp)

    def allocate(self, pc: bitarray, phr: bitarray, target: int):
        index = self.calc_index(pc, phr, self.history_length)
        tag = self.calc_tag(pc, phr, self.history_length)

        new_entry = ComponentEntry(bits=3, init_value=4, target=target,tag=tag)
        ways = self.entries.pop(index)
        ways = ways[:self.ways_number-1] + [new_entry]

        self.entries.insert(index, ways)

        # DEBUG
        self.occupied_entries[(index, tag)] = new_entry


class ConditionalTAGE:
    def __init__(self, history_max_length = 388):
        # self.btb = BTB(sets_number=pow(2,10), way)
        self.phr = PathHistoryRegister(history_max_length)

        # Create the tables:
        self.tables = [ConditionalBaseComponent(),
                       ConditionalTaggedComponent(history_length=68),
                       ConditionalTaggedComponent(history_length=132),
                       ConditionalTaggedComponent(history_length=388)]

    def predict(self, pc: int, phr: bitarray):
        phr = self.phr.register
        # Assuming that there was a BTB hit if we call this
        PC_LENGTH = 64
        pc_ba = int2ba(pc, length=PC_LENGTH, endian="little")

        # First, goto base table
#        if CONF.dbg_predictor:
#            print(f"PRED Base Table:")
        result = self.tables[0].predict(pc_ba)
        if result is None:
            self.allocate_in_base_table(pc, 0, None)
            result = self.tables[0].predict(pc_ba)

        prediction, confidence = result
        provider_index = 0

        # Now, iterate through all tagged predictors and use prediction in case of a tag match
        altpred_index = provider_index
        altpred = prediction
        altpred_confidence = confidence

        for i, tagged_component in enumerate(self.tables[1:]):
#            if CONF.dbg_predictor:
#                print(f"PRED Table {i+1}:")
            result = tagged_component.predict(pc_ba, phr)

            if result != None: # There was a tag match
                altpred = prediction
                altpred_index = provider_index
                altpred_confidence = confidence
                provider_index = i+1
                prediction = result[0]
                confidence = result[1]

        return (prediction, provider_index, confidence, altpred, altpred_index, altpred_confidence)

    def update(self, pc: int, phr: bitarray, target: int, actual_branch: int,
               prediction, provider_index, confidence, altpred, altpred_index, altpred_confidence):
        phr = self.phr.register
        PC_LENGTH = 64
        pc_ba = int2ba(pc, length=PC_LENGTH, endian="little")

#        if CONF.dbg_predictor:
#            print(f"UPDATE:")
#            print(f"TABLE IDX: {provider_index}")
#            print(f"PC: 0x{int(pc_ba.to01()[::-1], 2):x}")
#            print(f"ACTUAL_BRANCH: 0x{actual_branch:x}")
#            if target is not None:
#                print(f"TARGET: 0x{target:x}")
#            else:
#                print(f"TARGET IS NONE")
#            print(f"PHR: {int(phr.to01(), 2):0{PHR_MAX_SIZE}b}")
        # Always update the provider
        if provider_index == 0:
            self.tables[0].update(pc_ba, actual_branch, target)
        else:
            self.tables[provider_index].update(pc_ba, phr, actual_branch, target)

        # Low confidence => Also update altpred
        # if confidence==0 and altpred_index < provider_index:
        #     if altpred_index == 0:
        #         self.tables[0].update(pc_ba, actual_branch, target)
        #     else:
        #         # For now don't touch usefulness for altpred
        #         self.tables[altpred_index].update(pc_ba, self.phr.register, actual_branch, target, False)

        # Misprediction => Allocate a new entry if possible
        if prediction != actual_branch and provider_index < (len(self.tables) - 1):
#            if CONF.dbg_predictor:
#                print(f"UPDATE PROPAGATE (ALLOCATE):")
#                print(f"AT TABLE IDX: {provider_index+1}")
            self.tables[provider_index+1].allocate(pc_ba, phr, target)

    def allocate_in_base_table(self, pc: int, actual_branch: int, target: int):
        PC_LENGTH = 64
        pc_ba = int2ba(pc, length=PC_LENGTH, endian="little")

#        if CONF.dbg_predictor:
#            print(f"BASE ALLOCATE:")
#            print(f"PC: 0x{int(pc_ba.to01()[::-1], 2):x}")
#            print(f"ACTUAL_BRANCH: 0x{actual_branch:x}")
#            if target is not None:
#                print(f"TARGET: 0x{target:x}")
#            else:
#                print(f"TARGET IS NONE")

        self.tables[0].allocate(pc_ba, actual_branch, target)

    # def allocate_in_btb(self, pc: int, target_if_taken: int):
    #     PC_LENGTH = 64
    #     pc_ba = int2ba(pc, length=PC_LENGTH, endian="little")

    #     btb_entry = self.btb.search(pc_ba)
    #     if btb_entry and btb_entry.target_if_taken == target_if_taken:

    def update_phr(self, pc: int, target: int):
        self.phr.update(pc, target)

    # DEBUG
    def print_cbp(self, base_address: int):
        print("\n=== CBP State ===")
        if self.tables[0].occupied_entries != {}:
            print(">>> Base Component:")
            print(f"{'Index':>6} | {'Counter':>8} | {'Target':>8}")
            print("-" * 30)
            for index, entry in self.tables[0].occupied_entries.items():
                state = entry.counter.state
                target = entry.target
                state_str = f"{state}"
                print(f"{index:>6} | {state:>8} | {hex(target-base_address):>8}")
        else:
            print("No entries occupied in Base Component.")

        if len([table for table in self.tables[1:] if table.occupied_entries != {}]) != 0:
            print("\n>>> Tagged Components:")
            for i, tagged in enumerate(self.tables[1:]):
                if tagged.occupied_entries == {}:
                    continue
                print(f"\n-- Tagged Table {i+1} (History Length = {tagged.history_length}) --")
                print(f"{'Index':>6} | {'Tag':>10} | {'Counter':>8} | {'Target':>8}")
                print("-" * 38)
                for (index, tag), entry in tagged.occupied_entries.items():
                    state = entry.counter.state
                    state_str = f"{state}"
                    target = entry.target
                    print(f"{index:>6} | {tag:>10} | {state_str:>8} | {hex(target-base_address):>8}")
            print("\n")
        else:
            print("No entries occupied in Tagged Components.\n")


# ==========================================================================================
#
#                                       Indirect Branch Predictor
#
# ==========================================================================================

# Definition for Entry class used in IndirectTaggedTable
class Entry:
    def __init__(self, target, tag, type="indirect"):
        self.target = target
        self.tag = tag
        self.type = type
        # NOTE: not reproducible when having collisions on BTB


    def get_tag(self):
        return self.tag

    def get_state(self):
        return self.target

    def update_correct(self, actual_target):
        # Update the target address in the entry
        pass

    def update_incorrect(self, actual_target):
        # Update the target address in the entry
        self.target = actual_target

    #TODO: add check for a match (type and more...)

class IndirectBranchPredictor(BranchPredictor):
    def __init__(self, init_state_val, pht_size):
        super().__init__(init_state_val, pht_size)

        NUMBER_OF_BITS_TO_SAVE = 64
        self.pattern_history_table = [IndirectState(NUMBER_OF_BITS_TO_SAVE, init_state_val)
                                      for _ in range(pht_size)]
        offset = 0
        self.pht_numbits = math.frexp(pht_size)[1] - 1

        self.cut_pc = [self.pht_numbits + offset, offset]

    def predict(self, pc, actual_target):
        cutpc = get_from_bitrange(self.cut_pc, pc)  # NOTE: here it calculates pht entry

        prediction = self.prediction_method(cutpc, actual_target)
        return prediction

    @abstractmethod
    def prediction_method(self, cutpc, actual_target):
        pass

class IndirectOneLevel(IndirectBranchPredictor):
    def __init__(self, init_state_val, pht_size):
        super().__init__(init_state_val, pht_size)

    def prediction_method(self, cutpc, actual_target):
        pht_address = cutpc
        prediction = self.pattern_history_table[pht_address].get_state() # TODO: think if it's returning full address

        self.pattern_history_table[pht_address].update_prediction(actual_target)

        return prediction

class IndirectTaggedTable:
    #TODO: use tag for hash table
    def __init__(self, sets_number = pow(2,10), ways_number = 12, ip_based_only = False):

        #TODO: initialize index and tag function
        self.ways_number = ways_number
        self.table = [[] for _ in range(sets_number)]


        self.entry_class = Entry
        self.lowest = True # correct for BTB and T1 only

    def predict(self, pc: bitarray, phr : bitarray): # TODO: don't support type for now
        prediction = None
        index = self.calc_index(pc,phr)
        tag = self.calc_tag(pc, phr)
        for i in range(len(self.table[index])):
            if self.table[index][i].get_tag() == tag:
                prediction = self.table[index][i].get_state()
                # NOTE: update LRU
                self.table[index].insert(0, self.table[index].pop(i))
                break
        return prediction # NOTE: NOT necessarily the target address (but could be only partial)

    # Note: assuming that would be called only if needed total mis-prediction or if was the provider correct target
    def update(self, pc: bitarray, phr: bitarray, target: bitarray, prediction: bitarray):
            #TODO: maybe send just index+tag
            index = self.calc_index(pc,phr)
            if DEBUG_PRINT:
                print(f"when updating: {index}")
            tag = self.calc_tag(pc, phr)

            # was not a miss
            if prediction == target:
                # moved to first
                self.table[index][0].update_correct(target)
            elif prediction:
                self.table[index][0].update_incorrect(target)
            else:  # was a miss / not used.
                # allocate a new entry
                entry = self.entry_class(target, tag, type = "indirect") # TODO: update tag
                self.table[index] = [entry] + self.table[index][:self.ways_number-1]

    @abstractmethod
    def calc_index(self, pc:bitarray,phr:bitarray) -> int:
        pass

    @abstractmethod
    def calc_tag(self, pc:bitarray, phr:bitarray) -> int:
        pass
    #TODO: change mechanism of update to be on resolve only (?)

MAX_CONFIDENCE = 3 # for indirect tage
INITIAL_CONFIDENCE = 0 # for indirect tage
# Minimal definition for IndirectTAGEEntry to avoid NameError
class IndirectTAGEEntry(Entry):
    def __init__(self, target, tag, type="indirect", confidence=INITIAL_CONFIDENCE, max_confidence=MAX_CONFIDENCE):
        self.confidence = confidence
        self.max_confidence = max_confidence
        super().__init__(target, tag, type)

    def update_correct(self, actual_target):
        if self.confidence < self.max_confidence:
            self.confidence += 1

    def update_incorrect(self, actual_target):
        if self.confidence > 0:
            self.confidence -= 1
        else:
            self.target = actual_target


PHR_MAX_SIZE = 194 * 2
TAG_LENGTH = 11
INDEX_LENGTH = 9

def get_bit(x, i):
    if i == -1:
        return 0
    return x[i]

def build_bitarray_from_indexes(x, indices):
    arr = bitarray(len(indices), endian="little")
    for i, bit_idx in enumerate(indices):
        if bit_idx == -1:
            bit = 0
        else:
            bit = get_bit(x, bit_idx)
        arr[i] = bit
    return arr

class IndirectTageTaggedTable(IndirectTaggedTable):
    def __init__(self, sets_number = pow(2,9), ways_number = 12, history_length = 388):
        super().__init__(sets_number, ways_number, False)
        self.history_length = history_length
        self.entry_class = IndirectTAGEEntry #NOTE: really stupid solution but IDC:)

    #TODO: switch sides of pc & phr
    #TODO: put the implementation here

    #TODO: this could still be used as int
    @staticmethod
    def compute_ibp_tag(pc: bitarray, phr: bitarray, phr_slice_length:int = PHR_MAX_SIZE) -> int:
        """
        Compute the 11-bit IBP tag from a 388-bit PHR and 16-bit PC using folding logic.
        """
        tag = 0
        even_bits = bitarray(TAG_LENGTH,endian="little")
        odd_bits = bitarray(TAG_LENGTH,endian="little")

        #first one - 10 bits
        for iteration, i in enumerate(range(16,35, 2)):
            offset = TAG_LENGTH - 10
            bit_index = (iteration + offset) % TAG_LENGTH
            even_bits[bit_index] = get_bit(phr, i)
            odd_bits[bit_index] = get_bit(phr, i + 1)

        for iteration, i in enumerate(range(36, phr_slice_length-1, 2)):
            bit_index = iteration % TAG_LENGTH
            even_bits[bit_index] ^= get_bit(phr, i)
            odd_bits[bit_index] ^= get_bit(phr, i + 1)

        odd_bits = odd_bits[5:] + odd_bits[:5] # do a rotation

        lower_phr_bits = bitarray(TAG_LENGTH,endian="little")

        first = [4,6,8,0,2,12,14,-1,-1,10,-1]
        second = [15,-1,-1,11,13,3,5,7,9,1,-1]
        lower_phr_bits = build_bitarray_from_indexes(phr, first) ^ \
            build_bitarray_from_indexes(phr, second)

        pc_bits_indices = [10,12,14,6,8,9,11,13,15,7,-1]
        pc_bits = build_bitarray_from_indexes(pc, pc_bits_indices)

        tag = pc_bits ^ lower_phr_bits ^ even_bits ^ odd_bits
        if DEBUG_PRINT:
            print(f"Tag: {tag}")
        return ba2int(tag)

    @staticmethod
    def compute_ibp_index(pc: bitarray, phr: bitarray, phr_slice_length = PHR_MAX_SIZE) -> bitarray:
        """
        Compute the 11-bit IBP tag from a 384-bit PHR and 16-bit PC using folding logic.
        """
        #initialize the index bits to 0 in correct size
        even_bits = bitarray(INDEX_LENGTH, endian="little")
        odd_bits = bitarray(INDEX_LENGTH, endian="little")

        #first one - 6 bits
        offset = INDEX_LENGTH - 6
        for iteration, i in enumerate(range(16, 27, 2)):
            bit_index = (iteration+offset) % INDEX_LENGTH
            even_bits[bit_index] = get_bit(phr, i)
            odd_bits[bit_index] = get_bit(phr, i + 1)

        for iteration, i in enumerate(range(28, phr_slice_length-1, 2)): # for example range(28,387,2)
            bit_index = iteration % INDEX_LENGTH
            even_bits[bit_index] ^= get_bit(phr, i)
            odd_bits[bit_index] ^= get_bit(phr, i + 1)

        # NOTE: fixed after using Indirector code
        odd_bits = (odd_bits[4:] + odd_bits[:4])

        index = even_bits ^ odd_bits
        if DEBUG_PRINT:
            print(f"Index: {index}")
        return ba2int(index)

    def calc_index(self, pc:bitarray, phr:bitarray) -> int:
        return IndirectTageTaggedTable.compute_ibp_index(pc, phr, self.history_length)

    def calc_tag(self, pc: bitarray, phr: bitarray) -> int:
        return IndirectTageTaggedTable.compute_ibp_tag(pc, phr, self.history_length)

class IndirectBTB(IndirectTaggedTable):
    def __init__(self, sets_number = pow(2,10), ways_number = 12):
        super().__init__(sets_number, ways_number, True)

    #Indirector Table 1.
    def calc_index(self, pc:bitarray, phr:bitarray) -> int:
        return ba2int(pc[5:15])

    #Indirector Table 1.
    def calc_tag(self, pc:bitarray, phr:bitarray) -> int:
        return ba2int(pc[15:24] + pc[0:5]) #TODO: should it really be int

    def update(self, pc: bitarray, phr: bitarray, target: bitarray, prediction: bitarray):
        return super().update(pc, phr, target, prediction)


#TODO: implement indirect TAGE
#TODO: use PHR outside of the predictor (as global register -- maybe inside the uc)
class IndirectTAGEPredictor:
    def __init__(self, tables_history, sets_number, ways_number, history_max_length = 388):

        # Init tagged predictors
        self.tagged_predictors = []
        for history_length in tables_history:
            self.tagged_predictors.append(IndirectTageTaggedTable(sets_number, ways_number, history_length))

        # self.global_history_register = PathHistoryRegister(history_max_length)

        self.count = 0
        self.msb_flip = True

    def predict(self, pc: bitarray, phr: bitarray):
        predictions = []

        for i, table in enumerate(self.tagged_predictors):
            #NOTE: would also apply lru. (not the same as useful bits - but I cannot simulate that)
            if DEBUG_PRINT:
                print(f"Table {i} prediction")
            pred =self.tagged_predictors[i].predict(pc, phr)
            predictions.append(pred)


        #find predictions
        provider_index = -1
        altpred_index = -1
        altpred = None
        prediction = None
        for i in range(len(predictions)):
            if predictions[i] is not None:
                altpred = prediction
                altpred_index = provider_index
                provider_index = i
                prediction = predictions[i]
            #for now allocating new one in probability of 1/2:
        # TODO: update whether altpred was useful or not.
        return (prediction, provider_index, altpred, altpred_index)
        #use alt+pred to update usefulness

    def update(self, pc, phr, target, prediction, provider_idx, altpred, altpred_idx):
        if prediction == target:
            self.tagged_predictors[provider_idx].update(pc, phr, target, prediction)
        #there was an incorrect prediction
        elif prediction: # same as provider_idx != -1
            #update prvoider
            self.tagged_predictors[provider_idx].update(pc, phr, target, prediction)
            #update the provider

            if provider_idx != len(self.tagged_predictors) -1:
                #could allocate in higher table
                # if random.choice([0, 1]): # 50% chance to update --- RANDOM
                if True:
                    # new_table_idx = random.randint(provider_idx+1, len(self.tagged_predictors) - 1) ---RANDOM
                    new_table_idx = provider_idx + 1 # IMPORANT: currently just allocate in the above table
                    self.tagged_predictors[new_table_idx].update(pc, phr, target, None)
        #there was a miss
        else:
            # new_table_idx = random.randint(0, len(self.tagged_predictors) - 1) --- RANDOM
            new_table_idx = 0 #Important: currently just allocate in the first table
            self.tagged_predictors[new_table_idx].update(pc, phr, target, None)

        pass

#BTB + IBP
PC_LENGTH = 64
class IndirectCombinedPredictor:
    def __init__(self, history_max_length = 388):
        # Indirector Figure 10
        self.btb = IndirectBTB(pow(2,10), 12) #TODO: put in params
        self.phr = PathHistoryRegister(history_max_length)
        self.ibp = IndirectTAGEPredictor([68, 132, 388], pow(2,9), 2, history_max_length)
        self.last_prediction = {"chosen_prediction": None, "btb_prediction": None, "tage_prediction": None}
        #NOTE: save (chosen_prediction, btb_prediction, (tage_prediction, provider_index, altpred_index))


    def predict(self, pc: int, correct_target: int, phr: Optional[PathHistoryRegister] = None):

        if phr is None:
            phr = self.phr.register
        else:
            phr = phr.register
        pc = int2ba(pc, length=PC_LENGTH, endian='little') # tranlate to ba
        mark_high_bits = int2ba(0xFFFFFFFF00000000, length=PC_LENGTH, endian='little') # mask for high bits
        prediction_btb = self.btb.predict(pc, phr)
        prediction_tage, provider_idx, altpred, altpred_idx = self.ibp.predict(pc, phr)

        prediction_details = {}
        prediction_details["btb_prediction"] = prediction_btb
        prediction_details["tage_prediction"] = (prediction_tage, provider_idx, altpred, altpred_idx)

        final_prediction = None
        if prediction_btb and prediction_tage:
            #TODO: decide what to take -- maybe ST vs MT? (important - should mark if have been changed before) -- found to be not needed
            #NOTE TO CHECK: if it's ST, thus the prediction should be the same in both, so we can always handle it as BTB(if we don't take partial update into consideration)

            final_prediction = prediction_tage

            prediction_details["chosen_prediction"] = "tage"
        elif prediction_btb:

            # Indirector section 3.1.1: BTB saves only 32 bits
            final_prediction = (pc & mark_high_bits) | prediction_btb

            prediction_details["chosen_prediction"] = "btb"
        else:
            final_prediction = None
            prediction_details["chosen_prediction"] = "none"
            # NOTE: in that way, there is no prediction at all. (as not recognized as branch--can act like phantom branch = not-branch address that appear on BTB)

        #NOTE: according to Indirector Table 2.
        # elif prediction_tage:
            # final_prediction = prediction_tage
            # prediction_details["chosen_prediction"] = "tage"

        self.last_prediction = prediction_details
        printing = {}
        for k in self.last_prediction.keys():
            if "btb" in k:
                if self.last_prediction[k] is not None:
                    printing[k] = hex(ba2int(self.last_prediction[k]))
                else:
                    printing[k] = self.last_prediction[k]
            elif "tage" in k:
                if len(self.last_prediction[k]) > 0 and self.last_prediction[k][0] is not None:
                    printing[k] = (hex(ba2int(self.last_prediction[k][0])), self.last_prediction[k][1], self.last_prediction[k][2], self.last_prediction[k][3])
                else:
                    printing[k] = self.last_prediction[k]
            else:
                printing[k] = self.last_prediction[k]

        if DEBUG_PRINT:
            print(printing, f" PHR used: {phr}")

        if final_prediction is not None:
            return ba2int(final_prediction) #translate to int
        else:
            return final_prediction

    def update(self, pc: int, target: int, jump_to_following_instruction: bool = False, phr: Optional[PathHistoryRegister] = None):

        # NOTE: jump_to_following_instruction - is if the target of the branch is the following instruction
        if phr is None:
            phr = self.phr.register
        else:
            phr = phr.register

        pc = int2ba(pc, length=PC_LENGTH, endian='little')
        target = int2ba(target, length=PC_LENGTH, endian='little')
        btb_bits = int2ba(0xFFFFFFFF, length=PC_LENGTH, endian='little')

        #correct prediction
        if self.last_prediction["chosen_prediction"] == "tage" and self.last_prediction["tage_prediction"][0] == target:
            self.ibp.update(pc, phr, target, *self.last_prediction["tage_prediction"])
        elif self.last_prediction["chosen_prediction"] == "btb" and self.last_prediction["btb_prediction"] == (target & btb_bits):
            # NOTE: there is no info about confidence counter in BTB, also not supporting replacement policy
            pass
        else: #On misprediction update both BTB and IBP -- according to Observation 4 in BPRC paper.
            # Handling an edge case: Claim D.2 - updating BTB only
            if self.last_prediction["chosen_prediction"] == "none" and jump_to_following_instruction:
                self.btb.update(pc, phr, target & btb_bits, self.last_prediction["btb_prediction"])
            else:
                if self.last_prediction["tage_prediction"][0] != target:
                    self.ibp.update(pc, phr, target, *self.last_prediction["tage_prediction"])
                if self.last_prediction["btb_prediction"] != (target & btb_bits):
                    self.btb.update(pc, phr, target & btb_bits, self.last_prediction["btb_prediction"])

            #NOTE: don't update phr for supporting all kinds of branches

    def prediction_method(self, pc, target):
        self.predict(pc, target)
        self.update(pc, target)

    def update_phr(self, pc, target):
        self.phr.update(pc, target)

# partial predictor for checking BTB without using IBP (can do so with BHI_DIS_S enabled)
class IndirectBTBPredictor:
    def __init__(self, history_max_length = 388):
    # Indirector Figure 10
        # self.btb = BTB(pow(2,10), 12) #TODO: put in params
        self.btb = IndirectBTB(pow(2,10), 12) #TODO: put in params
        # self.phr = PathHistoryRegister(history_max_length)
        # self.ibp = IndirectTAGE([68, 132, 388], pow(2,9), 2, history_max_length)
        self.last_prediction = {"chosen_prediction": None, "btb_prediction": None}
        #NOTE: save (chosen_prediction, btb_prediction, (tage_prediction, provider_index, altpred_index))

    def predict(self, pc: int, correct_target: int):
        pc = int2ba(pc, length=PC_LENGTH, endian='little') # tranlate to ba
        mark_high_bits = int2ba(0xFFFFFFFF00000000, length=PC_LENGTH, endian='little') # mask for high bits

        prediction_btb = self.btb.predict(pc, bitarray(PHR_MAX_SIZE, endian='little')) #TODO: PHR is not used here, but used for compatibility with IndirectCombinedPredictor

        prediction_details = {}
        prediction_details["btb_prediction"] = prediction_btb

        final_prediction = None
        if prediction_btb:
            prediction_details["chosen_prediction"] = "btb"
            final_prediction = (pc & mark_high_bits) | prediction_btb
        else:
            final_prediction = None
            prediction_details["chosen_prediction"] = "none"
            # NOTE: in that way, there is no prediction at all. (as not recognized as branch--can act like phantom branch, but discarded for now).


        self.last_prediction = prediction_details
        printing = {}
        for k in self.last_prediction.keys():
            if "btb" in k:
                if self.last_prediction[k] is not None:
                    printing[k] = hex(ba2int(self.last_prediction[k]))
                else:
                    printing[k] = self.last_prediction[k]
            else:
                printing[k] = self.last_prediction[k]

        if final_prediction is not None:
            return ba2int(final_prediction) #translate to int
        else:
            return final_prediction


    def update(self, pc: int, target: int):
        pc = int2ba(pc, length=PC_LENGTH, endian='little')
        target = int2ba(target, length=PC_LENGTH, endian='little')
        btb_bits = int2ba(0xFFFFFFFF, length=PC_LENGTH, endian='little') # Stores only 32 lower bits of target address
        #decide how to update

        #correct prediction
        if self.last_prediction["chosen_prediction"] == "btb" and self.last_prediction["btb_prediction"] == (target & btb_bits):
            # Do not update BTB on correct prediction
            pass
        else: #On misprediction update both BTB and IBP -- according to Observation 4 in BPRC paper.
            if self.last_prediction["btb_prediction"] != (target & btb_bits):
                self.btb.update(pc, bitarray(PHR_MAX_SIZE, endian='little'), target & btb_bits, self.last_prediction["btb_prediction"]) # PHR is not needed here, but used for compatibility with IndirectCombinedPredictor

    def prediction_method(self, pc, target):
        self.predict(pc, target)
        self.update(pc, target)


    # def reset_indirect_instructions(self):
    #     """
    #     Reset the indirect instructions in the BTB.
    #     This is a placeholder for any specific reset logic needed for indirect instructions.
    #     """
    #     # Currently, this method does not perform any action, but can be extended in the future.
    #     for idx, set_idx in enumerate(self.btb.table):
    #         new_set = []
    #         for entry in set_idx:
    #             # Reset the entry if needed, e.g., by clearing the target or tag
    #             if entry.type == "indirect":
    #                 continue
    #             new_set.append(entry)
    #         self.btb.table[idx] = new_set


class RsbPredictor:
    def __init__(self, num_of_entries):
        self.stack = []
        #self.curr_size = 0
        self.max_size = num_of_entries

    # process ret - would update or not
    def predict_ret(self, pc: int, correct_target : int):
    # if empty
        if not self.stack:
            return None # here should go to btb
        return ba2int(self.stack[0])

    def update_call(self, pc :int, target: int):
        target = int2ba(target, length=PC_LENGTH, endian='little')
        self.stack.insert(0, target) # insert at the top of the stack

        # overflow
        if len(self.stack) >= self.max_size:
            self.stack.pop()

    def update_ret(self, pc: int, target: int):
        if self.stack:
            self.stack = self.stack[1:]


    def is_empty(self):
        return len(self.stack) == 0

    # TODO: don't support nested speculation(as won't update the first in stack)
