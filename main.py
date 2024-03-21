# Define the decimal digits
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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

# Function to add two decimal numbers
def decimal_add(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 + int2
    return integer_to_decimal(result)

# Function to subtract two decimal numbers
def decimal_subtract(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 - int2
    return integer_to_decimal(result)

# Function to multiply two decimal numbers
def decimal_multiply(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    result = int1 * int2
    return integer_to_decimal(result)

# Function to divide two decimal numbers
def decimal_divide(decimal1, decimal2):
    int1 = decimal_to_integer(decimal1)
    int2 = decimal_to_integer(decimal2)
    if int2 == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    result = int1 // int2
    return integer_to_decimal(result)

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

    def fetch_instruction(self):
        instruction = self.memory.store(self.program_counter)
        self.program_counter += 1
        return instruction

    def decode_instruction(self, instruction):
        parts = instruction.split()
        operation = parts[0]
        operands = parts[1:]
        return operation, operands

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
        elif operation in ["add", "subtract", "multiply", "divide", "and", "or", "xor", "not", "left_shift", "right_shift"]:
            operand1 = self.registers.store(operands[0])
            operand2 = self.registers.store(operands[1]) if len(operands) > 1 else None
            result = self.alu.execute(operation, operand1, operand2)
            self.registers.load(operands[-1], result)
        elif operation == "jump":
            address = decimal_to_integer(operands[0])
            self.program_counter = address
        elif operation == "jump_if_zero":
            address = decimal_to_integer(operands[0])
            if self.registers.store(operands[1]) == '0':
                self.program_counter = address
        elif operation == "jump_if_not_zero":
            address = decimal_to_integer(operands[0])
            if self.registers.store(operands[1]) != '0':
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
        else:
            raise ValueError(f"Invalid instruction: {operation}")

    def run_program(self):
        while True:
            instruction = self.fetch_instruction()
            if instruction == "halt":
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

# Example usage
memory_size = 100
program = [
    "load acc 5",
    "load r1 10",
    "add acc r1",
    "store acc 50",
    "output 50",
    "halt"
]

kernel = Kernel(memory_size)
kernel.load_program(program)
kernel.run()
