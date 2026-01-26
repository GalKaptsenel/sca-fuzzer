def create_file(tc_path: str, input_path: str, register_offset: int, output_path: str):
    # Read test case and input contents first
    with open(tc_path, "rb") as tcf:
        tc_data = tcf.read()
    with open(input_path, "rb") as inf:
        input_data = inf.read()

    # Compute sizes
    code_size = len(tc_data)
    mem_size = register_offset
    regs_size = len(input_data) - mem_size

    with open(output_path, "wb") as of:
        # Header
        of.write(b"RVZR")                               # magic
        of.write((0x0001).to_bytes(2, 'little'))        # version
        of.write((0x0002).to_bytes(2, 'little'))        # arch
        of.write((0x7).to_bytes(8, 'little'))          # flags

        # Sizes
        of.write(code_size.to_bytes(8, 'little'))       # code_size
        of.write(mem_size.to_bytes(8, 'little'))        # mem_size
        of.write(regs_size.to_bytes(8, 'little'))       # regs_size

        # Reserved field
        of.write((0).to_bytes(8, 'little'))

        # Append actual content
        of.write(tc_data)
        of.write(input_data)

create_file("tmp", "input.bin", 4096*2, "out_format_file")
