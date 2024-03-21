# Decimal Computer

This project is an implementation of a decimal-based computer architecture. It includes components such as an Arithmetic Logic Unit (ALU), registers, main memory, secondary memory, input/output devices, and a control unit. The computer supports various instructions and operations performed using decimal numbers.

## Features

- Arithmetic operations (addition, subtraction, multiplication, division)
- Logical operations (AND, OR, XOR, NOT)
- Shift operations (left shift, right shift)
- Conditional jumps (jump if zero, jump if not zero)
- Input/output operations
- Memory operations (load, store, load from secondary memory, store to secondary memory)
- Interrupt handling
- Pipelined execution
- Parallel execution using threads

## Components

### Arithmetic Logic Unit (ALU)

The ALU is responsible for performing arithmetic and logical operations on decimal numbers.

### Registers

Registers are used to store decimal values for various purposes, such as holding operands, results, and program counters.

### Main Memory

The main memory is a simulated memory space where instructions and data are stored. It supports loading and storing decimal values at specific addresses.

### Secondary Memory

The secondary memory represents additional storage space, such as a hard disk or flash drive, where data can be loaded from or stored to.

### Input/Output Devices

The input/output devices handle input and output operations, such as reading data from the terminal or writing data to printers or networks.

### Control Unit

The control unit is the heart of the computer. It fetches instructions from memory, decodes them, and executes them by coordinating the other components. It also handles interrupts and supports pipelined and parallel execution.

## Usage

1. Clone the repository or download the source code files.
2. Ensure you have Python installed on your system.
3. Open the `pc.py` file in a Python environment or editor.
4. Modify the `program` list to define the instructions you want to execute.
5. Create an instance of the `Kernel` class with the desired memory size.
6. Load the program into the kernel using the `load_program` method.
7. Run the program by calling the `run` method on the kernel instance.

## Example

```python
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
kernel.run()
``

## Future features:

- System Software and Operating System: Real computers have system software, including an operating system, device drivers, and utility programs, that manage the computer's resources, provide services to applications, and facilitate user interaction. Developing a basic operating system and system software components would make the decimal computer more complete and functional.
