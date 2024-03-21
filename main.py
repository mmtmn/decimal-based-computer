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

    def run_program(self):
        while True:
            # Check for interrupts
            self.check_interrupts()

            instruction = self.fetch_instruction()
            if instruction == "19":
                break
            operation, operands = self.decode_instruction(instruction)
            self.execute_instruction(operation, operands)
        

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
        self.io_devices = InputOutputDevices()
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

# Example usage
memory_size = 100

program = [
    "00 result_register 5.5",
    "00 temporary__register 10",
    "02 result_register temporary__register",
    "01 result_register 50",
    "16 50",
    "19"
]

kernel = Kernel(memory_size)
kernel.load_program(program)

# Register the interrupt handler
timer_interrupt = TimerInterrupt()
kernel.control_unit.register_interrupt(timer_interrupt)

# Store the interrupt handler code in memory
kernel.main_memory.load(6, "00 interrupt_register 1")  # Load the value 1 into the interrupt_register
kernel.main_memory.load(7, "16 51")  # Output the value stored at memory address 51
kernel.main_memory.load(8, "20")  # Return from the interrupt using the "RTI" instruction (opcode 20)

# Initialize memory location 51 with '0'
kernel.main_memory.load(51, '0')

# Run the program
kernel.run()