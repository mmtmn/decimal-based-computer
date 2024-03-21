import random
import struct
import threading

# Define the decimal digits
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

# Function to convert decimal to integer
def decimal_to_integer(decimal):
    integer = 0
    for digit in decimal:
        if digit not in digits:
            raise ValueError(f"Invalid decimal digit: {digit}")
        integer = integer * 10 + digits.index(digit)
    return integer

# Function to convert integer to decimal
def integer_to_decimal(integer):
    if integer == 0:
        return '0'
    decimal = ''
    while integer > 0:
        decimal = digits[integer % 10] + decimal
        integer //= 10
    return decimal

def float_to_decimal(float_num):
    integer_part = int(float_num)
    fractional_part = float_num - integer_part
    decimal = integer_to_decimal(integer_part)
    if fractional_part != 0:
        decimal += '.'
        while fractional_part != 0:
            fractional_part *= 10
            decimal += digits[int(fractional_part)]
            fractional_part -= int(fractional_part)
    return decimal

def decimal_add(decimal1, decimal2):
    float1 = decimal_to_float(decimal1)
    float2 = decimal_to_float(decimal2)
    result = float1 + float2
    return float_to_decimal(result)

def decimal_subtract(decimal1, decimal2):
    float1 = decimal_to_float(decimal1)
    float2 = decimal_to_float(decimal2)
    result = float1 - float2
    return float_to_decimal(result)

def decimal_multiply(decimal1, decimal2):
    float1 = decimal_to_float(decimal1)
    float2 = decimal_to_float(decimal2)
    result = float1 * float2
    return float_to_decimal(result)

def decimal_divide(decimal1, decimal2):
    float1 = decimal_to_float(decimal1)
    float2 = decimal_to_float(decimal2)
    if float2 == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    result = float1 / float2
    return float_to_decimal(result)

def decimal_to_float(decimal):
    if '.' in decimal:
        integer_part, fractional_part = decimal.split('.')
        integer = decimal_to_integer(integer_part)
        fraction = decimal_to_integer(fractional_part) / (10 ** len(fractional_part))
        return integer + fraction
    else:
        return decimal_to_integer(decimal)

# Function to perform logical AND on two decimal numbers
def decimal_and(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 & int2
    return integer_to_decimal(result)

# Function to perform logical OR on two decimal numbers
def decimal_or(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 | int2
    return integer_to_decimal(result)

# Function to perform logical XOR on two decimal numbers
def decimal_xor(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 ^ int2
    return integer_to_decimal(result)

# Function to perform logical NOT on a decimal number
def decimal_not(decimal):
    integer = decimal_to_integer(decimal)
    result = ~integer
    return integer_to_decimal(result)

# Function to perform left shift on a decimal number
def decimal_left_shift(decimal, shift):
    integer = decimal_to_integer(decimal)
    result = integer << shift
    return integer_to_decimal(result)

# Function to perform right shift on a decimal number
def decimal_right_shift(decimal, shift):
    integer = decimal_to_integer(decimal)
    result = integer >> shift
    return integer_to_decimal(result)

# Registers
class Registers:
    def __init__(self):
        self.registers = {}

    def load(self, register, value):
        self.registers[register] = value

    def store(self, register):
        return self.registers.get(register, '0')

# Arithmetic Logic Unit (ALU)
class ALU:
    def __init__(self):
        pass

    def execute(self, operation, operand1, operand2):
        if operation == "add":
            return decimal_add(operand1, operand2)
        elif operation == "subtract":
            return decimal_subtract(operand1, operand2)
        elif operation == "multiply":
            return decimal_multiply(operand1, operand2)
        elif operation == "divide":
            return decimal_divide(operand1, operand2)
        elif operation == "and":
            return decimal_and(operand1, operand2)
        elif operation == "or":
            return decimal_or(operand1, operand2)
        elif operation == "xor":
            return decimal_xor(operand1, operand2)
        elif operation == "not":
            return decimal_not(operand1)
        elif operation == "left_shift":
            return decimal_left_shift(operand1, decimal_to_integer(operand2))
        elif operation == "right_shift":
            return decimal_right_shift(operand1, decimal_to_integer(operand2))
        else:
            raise ValueError(f"Invalid ALU operation: {operation}")

# Main Memory
class MainMemory:
    def __init__(self, size):
        self.memory = ['0'] * size

    def load(self, address, value):
        if 0 <= address < len(self.memory):
            self.memory[address] = value
        else:
            raise IndexError(f"Invalid memory address: {address}")

    def store(self, address):
        if 0 <= address < len(self.memory):
            return self.memory[address]
        else:
            raise IndexError(f"Invalid memory address: {address}")

# Input/Output Devices
class InputOutputDevices:
    def __init__(self):
        pass

    def read_input(self):
        return input("Enter input: ")

    def write_output(self, data):
        print("Output:", data)

# Secondary Memory
class SecondaryMemory:
    def __init__(self):
        self.storage = {}

    def load(self, address, value):
        self.storage[address] = value

    def store(self, address):
        return self.storage.get(address, '0')

# Control Unit (CU)
class ControlUnit:
    def __init__(self, alu, registers, memory, io_devices, secondary_memory):
        self.alu = alu
        self.registers = registers
        self.memory = memory
        self.io_devices = io_devices
        self.secondary_memory = secondary_memory
        self.program_counter = 0
        self.interrupts = []
        self.interrupt_enabled = True

        # Pipeline stages
        self.fetch_stage = PipelineStage()
        self.decode_stage = PipelineStage()
        self.execute_stage = PipelineStage()
        self.write_back_stage = PipelineStage()

        # Instruction queue for parallel execution
        self.instruction_queue = []
        self.lock = threading.Lock()
        self.threads = []


    def register_interrupt(self, interrupt):
        self.interrupts.append(interrupt)

    def check_interrupts(self):
        for interrupt in self.interrupts:
            if interrupt.handler():
                self.handle_interrupt(interrupt)
                break

    def handle_interrupt(self, interrupt):
        if self.interrupt_enabled:
            # Save the current state
            self.registers.load("saved_pc", self.program_counter)
            
            # Set the program counter to the interrupt handler address
            self.program_counter = 6  # Assuming the interrupt handler code starts at memory address 6
            
            # Disable further interrupts
            self.interrupt_enabled = False

    def return_from_interrupt(self):
        # Restore the saved state
        self.program_counter = self.registers.store("saved_pc")
        
        # Enable interrupts
        self.interrupt_enabled = True

    def fetch_stage(self):
        while True:
            instruction = self.fetch_instruction()
            if instruction is None:
                break
            self.fetch_stage.add(instruction)

    def decode_stage(self):
        while True:
            instruction = self.fetch_stage.get()
            if instruction is None:
                break
            operation, operands = self.decode_instruction(instruction)
            self.decode_stage.add((operation, operands))

    def execute_stage(self):
        while True:
            operation_and_operands = self.decode_stage.get()
            if operation_and_operands is None:
                break
            operation, operands = operation_and_operands
            result = self.execute_instruction(operation, operands)
            self.execute_stage.add(result)

    def write_back_stage(self):
        while True:
            result = self.execute_stage.get()
            if result is None:
                break
            self.write_back(result)

    def run_program(self):
        while True:
            # Check for interrupts
            self.check_interrupts()

            # Pipeline stages
            self.fetch_stage()
            self.decode_stage()
            self.execute_stage()
            self.write_back_stage()

            # Parallel execution
            self.lock.acquire()
            for instruction in self.instruction_queue:
                thread = threading.Thread(target=self.execute_instruction, args=(instruction,))
                self.threads.append(thread)
                thread.start()
            self.instruction_queue.clear()
            self.lock.release()

            # Wait for worker threads to complete
            for thread in self.threads:
                thread.join()
            self.threads.clear()

            # Check for halt instruction
            if self.write_back_stage.buffer and self.write_back_stage.buffer[0] == "halt":
                break

    def execute_instruction(self, instruction):
        operation, operands = self.decode_instruction(instruction)
        # Execute the instruction using the existing execute_instruction method
        result = self.execute_instruction(operation, operands)
        # Add the result to the write-back stage
        self.write_back_stage.add(result)

    def write_back(self, result):
        operation, operands = result

        if operation == "load":
            register = operands[0]
            value = operands[1]
            self.registers.load(register, value)
        elif operation == "store":
            register = operands[0]
            address = decimal_to_integer(operands[1])
            value = self.registers.store(register)
            self.memory.load(address, value)
        elif operation in ["add", "subtract", "multiply", "divide", "and", "or", "xor"]:
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "not":
            register = operands[0]
            operand = self.registers.store(register)
            result = self.alu.execute(operation, operand, None)
            self.registers.load(register, result)
        elif operation in ["left_shift", "right_shift"]:
            register = operands[0]
            shift = operands[1]
            operand1 = self.registers.store(register)
            result = self.alu.execute(operation, operand1, shift)
            self.registers.load(register, result)
        elif operation == "jump":
            address = decimal_to_integer(operands[0])
            self.program_counter = address
        elif operation == "jump_if_zero":
            address = decimal_to_integer(operands[0])
            register = operands[1]
            if self.registers.store(register) == '0':
                self.program_counter = address
        elif operation == "jump_if_not_zero":
            address = decimal_to_integer(operands[0])
            register = operands[1]
            if self.registers.store(register) != '0':
                self.program_counter = address
        elif operation == "input":
            address = decimal_to_integer(operands[0])
            data = self.io_devices.read_input()
            self.memory.load(address, data)
        elif operation == "output":
            address = decimal_to_integer(operands[0])
            data = self.memory.store(address)
            self.io_devices.write_output(data)
        elif operation == "load_from_secondary":
            address_secondary = decimal_to_integer(operands[0])
            address_main = decimal_to_integer(operands[1])
            value = self.secondary_memory.store(address_secondary)
            self.memory.load(address_main, value)
        elif operation == "store_to_secondary":
            address_main = decimal_to_integer(operands[0])
            address_secondary = decimal_to_integer(operands[1])
            value = self.memory.store(address_main)
            self.secondary_memory.load(address_secondary, value)
        elif operation == "halt":
            pass  # Do nothing, the program will halt
        elif operation == "rti":
            self.return_from_interrupt()

    def __init__(self, alu, registers, memory, io_devices, secondary_memory):
        self.alu = alu
        self.registers = registers
        self.memory = memory
        self.io_devices = io_devices
        self.secondary_memory = secondary_memory
        self.program_counter = 0

    def fetch_instruction(self):
        instruction = self.memory.store(self.program_counter)
        self.program_counter += 1
        return instruction

    def decode_instruction(self, instruction):
        parts = instruction.split()
        opcode = parts[0]
        operands = parts[1:]

        valid_opcodes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

        if opcode not in valid_opcodes:        
            if opcode == "00":  # LOAD
                register = operands[0]
                value = operands[1]
                return ("load", [register, value])
            elif opcode == "01":  # STORE
                register = operands[0]
                address = operands[1]
                return ("store", [register, address])
            elif opcode == "02":  # ADD
                register1 = operands[0]
                register2 = operands[1]
                return ("add", [register1, register2])
            elif opcode == "03":  # SUBTRACT
                register1 = operands[0]
                register2 = operands[1]
                return ("subtract", [register1, register2])
            elif opcode == "04":  # MULTIPLY
                register1 = operands[0]
                register2 = operands[1]
                return ("multiply", [register1, register2])
            elif opcode == "05":  # DIVIDE
                register1 = operands[0]
                register2 = operands[1]
                return ("divide", [register1, register2])
            elif opcode == "06":  # AND
                register1 = operands[0]
                register2 = operands[1]
                return ("and", [register1, register2])
            elif opcode == "07":  # OR
                register1 = operands[0]
                register2 = operands[1]
                return ("or", [register1, register2])
            elif opcode == "08":  # XOR
                register1 = operands[0]
                register2 = operands[1]
                return ("xor", [register1, register2])
            elif opcode == "09":  # NOT
                register = operands[0]
                return ("not", [register])
            elif opcode == "10":  # LEFT_SHIFT
                register = operands[0]
                shift = operands[1]
                return ("left_shift", [register, shift])
            elif opcode == "11":  # RIGHT_SHIFT
                register = operands[0]
                shift = operands[1]
                return ("right_shift", [register, shift])
            elif opcode == "12":  # JUMP
                address = operands[0]
                return ("jump", [address])
            elif opcode == "13":  # JUMP_IF_ZERO
                address = operands[0]
                register = operands[1]
                return ("jump_if_zero", [address, register])
            elif opcode == "14":  # JUMP_IF_NOT_ZERO
                address = operands[0]
                register = operands[1]
                return ("jump_if_not_zero", [address, register])
            elif opcode == "15":  # INPUT
                address = operands[0]
                return ("input", [address])
            elif opcode == "16":  # OUTPUT
                address = operands[0]
                return ("output", [address])
            elif opcode == "17":  # LOAD_FROM_SECONDARY
                address_secondary = operands[0]
                address_main = operands[1]
                return ("load_from_secondary", [address_secondary, address_main])
            elif opcode == "18":  # STORE_TO_SECONDARY
                address_main = operands[0]
                address_secondary = operands[1]
                return ("store_to_secondary", [address_main, address_secondary])
            elif opcode == "19":  # HALT
                return ("halt", [])
            elif opcode == "rti":
                self.return_from_interrupt()
            else:
                raise ValueError(f"Invalid opcode: {opcode}")
        else:
            raise InvalidInstructionError(f"Invalid opcode: {opcode}")

class ControlUnit:
    def __init__(self, alu, registers, memory, io_devices, secondary_memory):
        self.alu = alu
        self.registers = registers
        self.memory = memory
        self.io_devices = io_devices
        self.secondary_memory = secondary_memory
        self.program_counter = 0
        self.interrupts = []
        self.interrupt_enabled = True

    def register_interrupt(self, interrupt):
        self.interrupts.append(interrupt)

    def check_interrupts(self):
        for interrupt in self.interrupts:
            if interrupt.handler():
                self.handle_interrupt(interrupt)
                break

    def handle_interrupt(self, interrupt):
        if self.interrupt_enabled:
            # Save the current state
            self.registers.load("saved_pc", self.program_counter)
            
            # Set the program counter to the interrupt handler address
            self.program_counter = 6
            
            # Disable further interrupts
            self.interrupt_enabled = False

    def return_from_interrupt(self):
        # Restore the saved state
        self.program_counter = self.registers.store("saved_pc")
        
        # Enable interrupts
        self.interrupt_enabled = True

    def fetch_instruction(self):
        instruction = self.memory.store(self.program_counter)
        self.program_counter += 1
        return instruction

    def decode_instruction(self, instruction):
        parts = instruction.split()
        opcode = parts[0]
        operands = parts[1:]
        
        if opcode == "00":  # LOAD
            register = operands[0]
            value = operands[1]
            return ("load", [register, value])
        elif opcode == "01":  # STORE
            register = operands[0]
            address = operands[1]
            return ("store", [register, address])
        elif opcode == "02":  # ADD
            register1 = operands[0]
            register2 = operands[1]
            return ("add", [register1, register2])
        elif opcode == "03":  # SUBTRACT
            register1 = operands[0]
            register2 = operands[1]
            return ("subtract", [register1, register2])
        elif opcode == "04":  # MULTIPLY
            register1 = operands[0]
            register2 = operands[1]
            return ("multiply", [register1, register2])
        elif opcode == "05":  # DIVIDE
            register1 = operands[0]
            register2 = operands[1]
            return ("divide", [register1, register2])
        elif opcode == "06":  # AND
            register1 = operands[0]
            register2 = operands[1]
            return ("and", [register1, register2])
        elif opcode == "07":  # OR
            register1 = operands[0]
            register2 = operands[1]
            return ("or", [register1, register2])
        elif opcode == "08":  # XOR
            register1 = operands[0]
            register2 = operands[1]
            return ("xor", [register1, register2])
        elif opcode == "09":  # NOT
            register = operands[0]
            return ("not", [register])
        elif opcode == "10":  # LEFT_SHIFT
            register = operands[0]
            shift = operands[1]
            return ("left_shift", [register, shift])
        elif opcode == "11":  # RIGHT_SHIFT
            register = operands[0]
            shift = operands[1]
            return ("right_shift", [register, shift])
        elif opcode == "12":  # JUMP
            address = operands[0]
            return ("jump", [address])
        elif opcode == "13":  # JUMP_IF_ZERO
            address = operands[0]
            register = operands[1]
            return ("jump_if_zero", [address, register])
        elif opcode == "14":  # JUMP_IF_NOT_ZERO
            address = operands[0]
            register = operands[1]
            return ("jump_if_not_zero", [address, register])
        elif opcode == "15":  # INPUT
            address = operands[0]
            return ("input", [address])
        elif opcode == "16":  # OUTPUT
            address = operands[0]
            return ("output", [address])
        elif opcode == "17":  # LOAD_FROM_SECONDARY
            address_secondary = operands[0]
            address_main = operands[1]
            return ("load_from_secondary", [address_secondary, address_main])
        elif opcode == "18":  # STORE_TO_SECONDARY
            address_main = operands[0]
            address_secondary = operands[1]
            return ("store_to_secondary", [address_main, address_secondary])
        elif opcode == "19":  # HALT
            return ("halt", [])
        elif opcode == "20":  # RTI
            return ("rti", [])
        else:
            raise ValueError(f"Invalid opcode: {opcode}")

    def execute_instruction(self, operation, operands):
        if operation == "load":
            register = operands[0]
            value = operands[1]
            self.registers.load(register, value)
        elif operation == "store":
            register = operands[0]
            address = decimal_to_integer(operands[1])
            value = self.registers.store(register)
            self.memory.load(address, value)
        elif operation == "add":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "subtract":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "multiply":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "divide":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "and":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "or":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "xor":
            register1 = operands[0]
            register2 = operands[1]
            operand1 = self.registers.store(register1)
            operand2 = self.registers.store(register2)
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(register1, result)
        elif operation == "not":
            register = operands[0]
            operand = self.registers.store(register)
            result = self.alu.execute(operation, operand, None)
            self.registers.load(register, result)
        elif operation == "left_shift":
            register = operands[0]
            shift = operands[1]
            operand1 = self.registers.store(register)
            result = self.alu.execute(operation, operand1, shift)
            self.registers.load(register, result)
        elif operation == "right_shift":
            register = operands[0]
            shift = operands[1]
            operand1 = self.registers.store(register)
            result = self.alu.execute(operation, operand1, shift)
            self.registers.load(register, result)
        elif operation == "jump":
            address = decimal_to_integer(operands[0])
            self.program_counter = address
        elif operation == "jump_if_zero":
            address = decimal_to_integer(operands[0])
            register = operands[1]
            if self.registers.store(register) == '0':
                self.program_counter = address
        elif operation == "jump_if_not_zero":
            address = decimal_to_integer(operands[0])
            register = operands[1]
            if self.registers.store(register) != '0':
                self.program_counter = address
        elif operation == "input":
            address = decimal_to_integer(operands[0])
            device_type = "terminal"  # Assuming you want to input from the terminal
            protocol = None  # No protocol needed for terminal input
            data = self.io_devices.read_input(device_type, protocol)
            self.memory.load(address, data)
        elif operation == "output":
            address = decimal_to_integer(operands[0])
            data = self.memory.store(address)
            # Provide the device_type and protocol as additional arguments
            device_type = "terminal"  # Assuming you want to output to the terminal
            protocol = None  # No protocol needed for terminal output
            self.io_devices.write_output(data, device_type, protocol)
        elif operation == "load_from_secondary":
            address_secondary = decimal_to_integer(operands[0])
            address_main = decimal_to_integer(operands[1])
            value = self.secondary_memory.store(address_secondary)
            self.memory.load(address_main, value)
        elif operation == "store_to_secondary":
            address_main = decimal_to_integer(operands[0])
            address_secondary = decimal_to_integer(operands[1])
            value = self.memory.store(address_main)
            self.secondary_memory.load(address_secondary, value)
        elif operation == "halt":
            pass  # Do nothing, the program will halt
        elif operation == "rti":
            self.return_from_interrupt()
        else:
            raise ValueError(f"Invalid instruction: {operation}")

    def run_program(self):
        while True:
            # Check for interrupts
            self.check_interrupts()

            instruction = self.fetch_instruction()
            if instruction == "19":
                break
            operation, operands = self.decode_instruction(instruction)
            self.execute_instruction(operation, operands)
# Kernel
class Kernel:
    def __init__(self, memory_size):
        self.registers = Registers()
        self.alu = ALU()
        self.main_memory = MainMemory(memory_size)
        self.io_devices = io_devices  # Use the io_devices instance
        self.secondary_memory = SecondaryMemory()
        self.control_unit = ControlUnit(self.alu, self.registers, self.main_memory, self.io_devices, self.secondary_memory)
        
    def load_program(self, program):
        for address, instruction in enumerate(program):
            self.main_memory.load(address, instruction)

    def run(self):
        self.control_unit.run_program()

class Interrupt:
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler

# Define an interrupt handler
def timer_interrupt_handler():
    # Check if the timer interrupt condition is met
    # For example, let's assume the interrupt is triggered every 5 instructions
    if kernel.control_unit.program_counter % 5 == 0:
        print("Timer interrupt triggered!")
        # Store a different value in memory location 51
        kernel.main_memory.load(51, '1')
        # Jump to the interrupt handler address
        kernel.control_unit.handle_interrupt(timer_interrupt)
        return True  # Return True if the interrupt should be handled
    return False  # Return False if the interrupt should not be handled


# Define an interrupt handler
class TimerInterrupt:
    def __init__(self):
        self.is_handling_interrupt = False

    def handler(self):
        # Check if the timer interrupt condition is met
        # For example, let's assume the interrupt is triggered every 5 instructions
        if kernel.control_unit.program_counter % 5 == 0 and not self.is_handling_interrupt:
            print("Timer interrupt triggered!")
            # Store a different value in memory location 51
            kernel.main_memory.load(51, '1')
            # Jump to the interrupt handler address
            kernel.control_unit.handle_interrupt(self)
            self.is_handling_interrupt = True
            return True  # Return True if the interrupt should be handled
        return False  # Return False if the interrupt should not be handled
    
class MemoryAccessViolation(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class PageFaultException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class MemoryManagementUnit:
    def __init__(self, page_table, segment_table, physical_memory_size):
        self.page_table = page_table
        self.segment_table = segment_table
        self.page_size = 4096  # Assuming a page size of 4 KB
        self.physical_memory = ['0'] * physical_memory_size
        self.free_frames = list(range(physical_memory_size // self.page_size))

    def translate_address(self, virtual_address):
        segment_selector, offset = self.split_address(virtual_address)
        segment_base, segment_limit = self.segment_table[segment_selector]

        if offset < segment_limit:
            physical_address = self.translate_virtual_to_physical(segment_base, offset)
            return physical_address
        else:
            raise MemoryAccessViolation("Offset exceeded segment limit")

    def translate_virtual_to_physical(self, segment_base, offset):
        virtual_page_number = offset // self.page_size
        page_table_entry = self.page_table[segment_base + virtual_page_number]

        if page_table_entry is None:
            # Page fault handling
            self.handle_page_fault(segment_base, virtual_page_number)
            page_table_entry = self.page_table[segment_base + virtual_page_number]

        physical_page_frame = page_table_entry
        page_offset = offset % self.page_size
        physical_address = (physical_page_frame * self.page_size) + page_offset
        return physical_address

    def handle_page_fault(self, segment_base, virtual_page_number):
        if not self.free_frames:
            raise PageFaultException("No free physical frames available")

        # Allocate a new physical page frame
        physical_page_frame = self.free_frames.pop()

        # Update the page table entry
        self.page_table[segment_base + virtual_page_number] = physical_page_frame

    def split_address(self, virtual_address):
        segment_selector = virtual_address[:2]  # Assuming segment selector is the first two digits
        offset = decimal_to_integer(virtual_address[2:])
        return segment_selector, offset

# Parallel Port Protocol
class ParallelPortProtocol:
    def __init__(self, port_address):
        self.port_address = port_address
        self.data_register = 0
        self.status_register = 0
        self.control_register = 0

    def send_data(self, data):
        # Convert decimal data to integer
        data_integer = decimal_to_integer(data)

        # Set the data register with the data to be sent
        self.data_register = data_integer

        # Perform any necessary control register operations
        self.control_register |= 0x04  # Set the "Data Available" bit

        # Wait for the device to be ready
        while not self.status_register & 0x01:
            pass

        # Trigger the data transmission
        self.control_register |= 0x01  # Set the "Transmit" bit
        self.control_register &= ~0x04  # Clear the "Data Available" bit

    def receive_data(self):
        # Wait for data to be available
        while not self.status_register & 0x08:
            pass

        # Read the data from the data register
        data_integer = self.data_register

        # Convert the integer data to decimal
        data_decimal = integer_to_decimal(data_integer)

        # Perform any necessary control register operations
        self.control_register |= 0x08  # Set the "Data Received" bit

        return data_decimal

    def set_status_register(self, value):
        self.status_register = value

    def get_status_register(self):
        return self.status_register

    def set_control_register(self, value):
        self.control_register = value

    def get_control_register(self):
        return self.control_register
    def receive_data(self):
        # Wait for data to be available
        while not self.status_register & 0x02:
            pass

        # Read the data from the data register
        data_integer = self.data_register

        # Convert the integer data to decimal
        data_decimal = integer_to_decimal(data_integer)

        return data_decimal

# Serial Port Protocol
class SerialPortProtocol:
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate
        self.data_buffer = []
        self.receive_buffer = []

    def send_data(self, data):
        # Convert decimal data to integer
        data_integer = decimal_to_integer(data)

        # Convert the integer data to a sequence of bits
        bit_sequence = []
        for _ in range(8):
            bit_sequence.append(data_integer & 0x01)
            data_integer >>= 1

        # Add start and stop bits
        bit_sequence = [0] + bit_sequence + [1]

        # Transmit the bit sequence over the serial port
        for bit in bit_sequence:
            # Send the bit at the specified baud rate
            # ...
            pass

    def receive_data(self):
        # Check if there is data in the receive buffer
        if self.receive_buffer:
            data_bits = self.receive_buffer.pop(0)

            # Remove start and stop bits
            data_bits = data_bits[1:-1]

            # Convert the bit sequence to an integer
            data_integer = 0
            for bit in data_bits:
                data_integer = (data_integer << 1) | bit

            # Convert the integer data to decimal
            data_decimal = integer_to_decimal(data_integer)

            return data_decimal

        return '0'

# Printer Driver
class PrinterDriver:
    def __init__(self, port, protocol):
        self.port = port
        self.protocol = protocol

    def print_data(self, data):
        # Convert decimal data to integer
        data_integer = decimal_to_integer(data)

        # Convert the integer data to a sequence of bits
        bit_sequence = []
        for _ in range(8):
            bit_sequence.append(data_integer & 0x01)
            data_integer >>= 1

        # Send the bit sequence to the printer using the specified protocol
        self.protocol.send_data(bit_sequence)

# Network Driver
class NetworkDriver:
    def __init__(self, interface, protocol):
        self.interface = interface
        self.protocol = protocol

    def send_data(self, data, destination):
        # Convert decimal data to integer
        data_integer = decimal_to_integer(data)

        # Convert the integer data to a sequence of bits
        bit_sequence = []
        for _ in range(8):
            bit_sequence.append(data_integer & 0x01)
            data_integer >>= 1

        # Encapsulate the bit sequence with the destination address and any necessary headers
        packet = self.protocol.encapsulate(bit_sequence, destination)

        # Transmit the packet over the network interface
        self.interface.send(packet)

    def receive_data(self):
        # Receive a packet from the network interface
        packet = self.interface.receive()

        # Extract the bit sequence from the packet
        bit_sequence = self.protocol.extract(packet)

        # Convert the bit sequence to an integer
        data_integer = 0
        for bit in bit_sequence:
            data_integer = (data_integer << 1) | bit

        # Convert the integer data to decimal
        data_decimal = integer_to_decimal(data_integer)

        return data_decimal

# Network Protocol (e.g., Ethernet)
class EthernetProtocol:
    def __init__(self, src_mac, dst_mac):
        self.src_mac = src_mac  # Source MAC address
        self.dst_mac = dst_mac  # Destination MAC address
        self.ethernet_header_format = "!6s6s2s"  # Ethernet header format

    def encapsulate(self, data, destination):
        # Convert the destination address to bytes
        dst_mac_bytes = self.convert_decimal_to_bytes(destination, 6)

        # Create the Ethernet header
        ethernet_header = struct.pack(
            self.ethernet_header_format,
            self.convert_decimal_to_bytes(self.src_mac, 6),
            dst_mac_bytes,
            b"\x08\x00"  # Ethernet type (IP)
        )

        # Combine the Ethernet header and data
        packet = ethernet_header + self.convert_decimal_to_bytes(data, len(data))

        return packet

    def extract(self, packet):
        # Extract the Ethernet header
        ethernet_header_length = struct.calcsize(self.ethernet_header_format)
        header = packet[:ethernet_header_length]
        src_mac, dst_mac, ethernet_type = struct.unpack(self.ethernet_header_format, header)

        # Extract the data from the packet
        data_bytes = packet[ethernet_header_length:]

        # Convert the data bytes to decimal
        data_decimal = self.convert_bytes_to_decimal(data_bytes)

        return data_decimal

    def convert_decimal_to_bytes(self, decimal_string, num_bytes):
        decimal_value = decimal_to_integer(decimal_string)
        byte_list = decimal_value.to_bytes(num_bytes, byteorder='big')
        return byte_list

    def convert_bytes_to_decimal(self, byte_list):
        decimal_value = int.from_bytes(byte_list, byteorder='big')
        decimal_string = integer_to_decimal(decimal_value)
        return decimal_string

# Device Manager
class DeviceManager:
    def __init__(self):
        self.devices = {}

    def register_device(self, device_type, device_driver):
        self.devices[device_type] = device_driver

    def get_device_driver(self, device_type):
        return self.devices.get(device_type, None)

# Input/Output Devices
class InputOutputDevices:
    def __init__(self, device_manager):
        self.device_manager = device_manager

    def read_input(self, device_type, protocol):
        device_driver = self.device_manager.get_device_driver(device_type)
        if device_driver:
            return device_driver.receive_data(protocol)
        else:
            raise ValueError(f"Invalid device type: {device_type}")

    def write_output(self, data, device_type, protocol):
        device_driver = self.device_manager.get_device_driver(device_type)
        if device_driver:
            device_driver.send_data(data, protocol)
        else:
            raise ValueError(f"Invalid device type: {device_type}")

class PipelineStage:
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()

    def add(self, item):
        self.lock.acquire()
        self.buffer.append(item)
        self.lock.release()

    def get(self):
        self.lock.acquire()
        if self.buffer:
            item = self.buffer.pop(0)
            self.lock.release()
            return item
        self.lock.release()
        return None

class TerminalDriver:
    def __init__(self):
        pass

    def send_data(self, data, protocol=None):
        print("Output:", data)

    def receive_data(self, protocol=None):
        return input("Enter input: ")

class InvalidInstructionError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

# Create an instance of the DeviceManager
device_manager = DeviceManager()

# Instantiate the InputOutputDevices with the device_manager
io_devices = InputOutputDevices(device_manager)

# Create an instance of the TerminalDriver
terminal_driver = TerminalDriver()

# Register the terminal driver with the DeviceManager
device_manager.register_device("terminal", terminal_driver)



# # Example usage
# memory_size = 100

# program = [
#     "00 result_register 5.5",
#     "00 temporary__register 10",
#     "02 result_register temporary__register",
#     "01 result_register 50",
#     "16 50",
#     "19"
# ]

# kernel = Kernel(memory_size)
# kernel.load_program(program)

# # Register the interrupt handler
# timer_interrupt = TimerInterrupt()
# kernel.control_unit.register_interrupt(timer_interrupt)

# # Store the interrupt handler code in memory
# kernel.main_memory.load(6, "00 interrupt_register 1")  # Load the value 1 into the interrupt_register
# kernel.main_memory.load(7, "16 51")  # Output the value stored at memory address 51
# kernel.main_memory.load(8, "20")  # Return from the interrupt using the "RTI" instruction (opcode 20)

# # Initialize memory location 51 with '0'
# kernel.main_memory.load(51, '0')

# # Run the program
# kernel.run()