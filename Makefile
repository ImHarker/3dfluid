CPP = g++ 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
FLAGS = -Wall -O3 -pg -funroll-loops
all:
	$(CPP) $(SRCS) $(FLAGS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
