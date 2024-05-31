PHONY: build

build:
	gcc src/knn.c -o bin/knn


generate-data:
	python ./scripts/generate_points.py --n 1000 --dim 512 --out ./data/dataset-1K-512.txt