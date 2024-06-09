#!/bin/bash

timestamp=$(date +%s)

ncu --print-details all ./bin/main_cu > "profiling-report-${timestamp}.txt"