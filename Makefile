CPP = nvcc
SRCS = main.cu fluid_solver.cu EventManager.cpp
FLAGS = -O3 -lcuda -lm -arch=sm_35 -use_fast_math -Xptxas="-dlcm=ca"
FLAGS_PROFILE = -pg -g -fno-omit-frame-pointer

THREADS = 16

TARGET = fluid_sim

all:
	@echo Building Release Version
	$(CPP) $(SRCS) $(FLAGS) -o $(TARGET)
	@echo "Build complete."

run: all
	sbatch run.sh

clean:
	@echo Cleaning up...
	@rm $(TARGET)
	@echo Done.
