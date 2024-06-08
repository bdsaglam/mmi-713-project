PHONY: build-cpu build-gpu build generate-data run-cpu run-gpu

build-cpu:
	mkdir -p bin/
	g++ src/main.cpp -o bin/main_cpp -lm -std=c++11
	g++ src/test_sorting.cpp -o bin/test_sorting_cpp -lm

build-gpu:
	mkdir -p bin/
	nvcc src/main.cu -o bin/main_cu

build: build-cpu build-gpu

test: build-cpu
	bin/test_sorting_cpp

run-cpu: build-cpu
	bin/main_cpp $(ARGS)

run-gpu: build-gpu
	bin/main_cu $(ARGS)

generate-data:
	python ./scripts/generate_points.py --n 1000 --dim 512 --out ./data/dataset-1K-512.txt

debug: 
	nvcc src/main2.cu -o bin/main2_cu && bin/main2_cu 100