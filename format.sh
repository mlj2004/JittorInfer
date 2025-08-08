#!/usr/bin/env bash

for name in $(find common examples ggml src tests/ -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \))
do
	echo Formatting $name
	clang-format-18 -i $name
done
