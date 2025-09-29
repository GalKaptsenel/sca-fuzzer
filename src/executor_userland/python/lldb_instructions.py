import lldb
import re

def get_registers():

    # Get the list of registers from LLDB
    regs = lldb.debugger.GetSelectedTarget().process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
    reg_values = {}
    for reg_set in regs:
        reg_values[reg_set.GetName()] = {}
        for reg in reg_set:
            reg_values[reg_set.GetName()][reg.GetName()] = reg.GetValue()
    return reg_values


def step_and_print(debugger, command, result, internal_dict):
    try:
        count = int(command.strip())
    except:
        result.PutCString("Usage: stepn <count>")
        return
    
    for _ in range(count):
        debugger.HandleCommand("disassemble --pc --count 1")
        debugger.HandleCommand("thread step-inst")

def step_until_exit(debugger, command, result, internal_dict):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()

    while process and process.GetState() == lldb.eStateStopped:
        debugger.HandleCommand("disassemble --pc --count 1")
        debugger.HandleCommand("thread step-inst")
        state = process.GetState()
        if state == lldb.eStateExited:
            result.PutCString("✅ Process exited.")
            break
        elif state == lldb.eStateCrashed:
            result.PutCString("💥 Process crashed.")
            break
        elif state == lldb.eStateStopped:
            # This could be a breakpoint, single-step, etc.
            thread = process.GetSelectedThread()
            stop_reason = thread.GetStopReason()
            if stop_reason == lldb.eStopReasonBreakpoint:
                result.PutCString("⛔ Hit breakpoint.")
                break

def _step_cfg_until(debugger, command, result, internal_dict, stop_cond):
    def is_branch_taken(pc_before, pc_after):
        if mnemonic == 'b':
            return True

        return pc_after != pc_before + 4

    def extract_registers(operands):
        # Simple ARM64 register pattern, e.g., x0, x1, w2, sp
        return re.findall(r'\b(x[0-9]+|w[0-9]+|sp|lr|fp)\b', operands.lower())

    def print_registers(label: str):
        def print_registers_table(registers, label=""):
            print(f"\n🔍 Registers {label}:\n" + "="*40)

            for set_name, reg_set in registers.items():
                print(f"\n📦 {set_name}:\n" + "-"*40)

                reg_items = list(reg_set.items())
                row = ""
                for i, (reg_name, reg_val) in enumerate(reg_items):
                    reg_str = f"{reg_name}: {reg_val or 'N/A'}".ljust(30)
                    row += reg_str
                
                    if (i + 1) % 4 == 0:
                        print(row)
                        row = ""
                if row:
                    print(row)

            print("="*40 + "\n")

        print_registers_table(get_registers(), label)

    def step_instruction_silent():
        result = lldb.SBCommandReturnObject()
        ci = lldb.debugger.GetCommandInterpreter()
        ci.HandleCommand("si", result)
        
        if result.Succeeded():
            return True
        else:
            print("Step failed:", result.GetError())
            return False

    def parse_arm64_mem_operand(operand_str):
        match = re.search(r'\[(.*?)\]', operand_str)
        if not match:
            return None
    
        contents = match.group(1).strip()
    
        # Replace registers with $reg for LLDB
        contents = re.sub(r'\b([wx]\d+|sp|xzr|wzr)\b', r'$\1', contents)
    
        # Handle shift: lsl #2 → << 2
        contents = re.sub(r'lsl\s+#?(\d+)', r'<< \1', contents, flags=re.IGNORECASE)
    
        # Handle shift: uxtx #2 → & 0xFFFFFFFFFFFFFFFF << 2 (just << 2 is usually fine)
        contents = re.sub(r'(uxtw|sxtx|sxtw)\s*#?(\d+)', r'<< \2', contents, flags=re.IGNORECASE)
    
        # Remove optional signs
        contents = contents.replace('#', '')
    
        # Remove type extensions (we assume full register values in LLDB)
        contents = re.sub(r'(uxtw|sxtx|sxtw)', '', contents, flags=re.IGNORECASE)
    
        # Replace commas with + (ARM syntax) but preserve order
        contents = re.sub(r'\s*,\s*', ' + ', contents)

        return contents.strip()

    def handle_memory(memory_operand, frame, target):
        effective_address_raw = frame.EvaluateExpression(memory_operand)

        if effective_address_raw.IsValid() and effective_address_raw.GetError().Success():

            effective_address = effective_address_raw.GetValueAsUnsigned()
            print(f'\tEffective Address Accessed: 0x{effective_address:x}')

            error = lldb.SBError()
            mem = target.ReadMemory(lldb.SBAddress(effective_address, target), 8, error)

            if not error.Success():
                print(f'Failed to read memory: {error.GetCString()}')

            byte_str = " ".join(f'{b:02x}' for b in mem)
            print(f'\t8-byte content before instruction execution {byte_str}')



    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()

    while process and process.GetState() == lldb.eStateStopped:
        frame = thread.GetSelectedFrame()
        pc_before = frame.GetPC()

        inst_list = target.ReadInstructions(frame.GetPCAddress(), 1)
        if not inst_list or len(inst_list) == 0:
            print("⚠️  Could not read instruction at PC.")
            break

        inst = inst_list[0]
        mnemonic = inst.GetMnemonic(target).lower()
        operands = inst.GetOperands(target)

        stream = lldb.SBStream()
        inst.GetDescription(stream)
        print(f'\n➡️  {stream.GetData()}')

        memory_operand = parse_arm64_mem_operand(operands)
        if(memory_operand):
            handle_memory(memory_operand, frame, target)

        if stop_cond(mnemonic):
            break

        is_cond_branch = mnemonic.startswith("b.") and len(mnemonic) == 4
        if is_cond_branch:
            try:
                nzcv_val = frame.FindRegister("nzcv").GetValueAsUnsigned()
                n = (nzcv_val >> 31) & 1
                z = (nzcv_val >> 30) & 1
                c = (nzcv_val >> 29) & 1
                v = (nzcv_val >> 28) & 1
                print(f"   🧪 NZCV = N:{n} Z:{z} C:{c} V:{v}")
            except:
                print("   ⚠️  Could not read NZCV register.")

       # Step instruction
        step_instruction_silent()

        print_registers("After Instruction")

        # Refresh PC
        frame = thread.GetSelectedFrame()
        pc_after = frame.GetPC()

        if inst.DoesBranch():
            if is_branch_taken(pc_before, pc_after):
                print("↪️  Branch taken")
            else:
                print("⏩ Branch not taken")

            if mnemonic in ["cbz", "cbnz", "tbnz", "tbz"]:
                regs = extract_registers(operands)
                for reg in regs:
                    try:
                        val = frame.FindRegister(reg).GetValue()
                        print(f"   🧪 {reg} = {val}")
                    except:
                        print(f"   ⚠️  Could not read value of {reg}")
            
        # Exit if process ends
        state = process.GetState()
        if state == lldb.eStateExited:
            print("\n✅ Process exited.")
            break
        elif state == lldb.eStateCrashed:
            print("\n💥 Process crashed.")
            break
        elif state == lldb.eStateStopped:
            stop_reason = thread.GetStopReason()
            if stop_reason == lldb.eStopReasonBreakpoint:
                print("\n⛔ Hit breakpoint.")
                break

def step_cfg_until_ret(debugger, command, result, internal_dict):
    print("🔁 Tracing instructions until `ret` is reached...\n")
    _step_cfg_until(debugger, command, result, internal_dict, lambda m: m.startswith('ret'))
    print("\n🏁 Hit `ret` instruction — stopping.") 

def step_cfg(debugger, command, result, internal_dict):
    _step_cfg_until(debugger, command, result, internal_dict, lambda _: False)

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f stepn.step_and_print stepn')
    debugger.HandleCommand('command script add -f stepn.step_until_exit step_until_exit')
    debugger.HandleCommand('command script add -f stepn.step_cfg step_cfg')
    debugger.HandleCommand('command script add -f stepn.step_cfg_until_ret step_cfg_until_ret')
    print("✅ Commands loaded: 'stepn', 'step_until_exit', 'step_cfg', 'step_cfg_until_ret'")

