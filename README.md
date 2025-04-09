# About

This repository contains CUDA programming examples and implementations. These programs are part of my learning journey with CUDA programming. This repository includes various CUDA programs that demonstrate parallel computing concepts using NVIDIA's CUDA platform. The programs cover fundamental operations like vector addition and more complex parallel algorithms.

## Reference

I heavily referred one of the classic books:

**Programming Massively Parallel Processors: A Hands-on Approach** by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj

to get a grasp of CUDA fundamentals and more advanced topics.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- NVCC (NVIDIA CUDA Compiler)
- C++ compiler

## Example: building and running

To build the vector addition program:
```bash
cd vector_add
nvcc main.cpp vector_add.cu -o vector_add
./vector_add
```

## License
MIT License 