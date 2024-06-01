PHONY: build-cpu build-gpu build generate-data

build-cpu:
	mkdir -p bin/
	gcc src/knn.c -o bin/knnc -lm

build-gpu:
	mkdir -p bin/
	nvcc src/knn.cu -o bin/knncu

build: build-cpu build-gpu

generate-data:
	python ./scripts/generate_points.py --n 1000 --dim 512 --out ./data/dataset-1K-512.txt


run-gpu: build-gpu
	bin/knncu