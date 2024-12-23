CPP = nvcc
SRCS = main.cpp fluid_solver.cu EventManager.cpp
FLAGS = -O3 -lcuda -lm -arch=sm_35
FLAGS_PROFILE = -pg -g -fno-omit-frame-pointer

THREADS = 16

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
