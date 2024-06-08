PHONY: build-cpu build-gpu build generate-data run-cpu run-gpu

build-cpu:
	mkdir -p bin/
	g++ src/main.cpp -o bin/main_cpp -lm -std=c++11
	g++ src/test_sorting.cpp -o bin/test_sorting_cpp -lm

build-gpu:
	mkdir -p bin/
	nvcc src/knn.cu -o bin/knn_cu

build: build-cpu build-gpu

generate-data:
	python ./scripts/generate_points.py --n 1000 --dim 512 --out ./data/dataset-1K-512.txt

run-cpu: build-cpu
	bin/main_cpp $(ARGS)

run-gpu: build-gpu
	bin/knn_cu $(ARGS)

test: build-cpu
	bin/test_sorting_cpp
