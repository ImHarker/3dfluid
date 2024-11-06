CPP = g++ 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
FLAGS = -Wall -Ofast -funroll-loops -ftree-vectorize -mtune=native -march=native -fopenmp -g -fno-omit-frame-pointer
FLAGS_PROFILE = -pg 

THREADS = 40

TARGET = fluid_sim

all:
	@echo Building Release Version
	$(CPP) $(SRCS) $(FLAGS) -o $(TARGET)
		@echo "Build complete."

profile:
	@echo Building Profiling Version
	$(CPP) $(SRCS) $(FLAGS) $(FLAGS_PROFILE) -o $(TARGET)
	@echo "Profiling build complete."

clean:
	@echo Cleaning up...
	@rm $(TARGET)
	@echo Done.

runseq:
	@echo Running sequential...
	export OMP_NUM_THREADS=1; ./$(TARGET)

runpar:
	@echo Running parallel...
	export OMP_NUM_THREADS=$(THREADS); ./$(TARGET)
