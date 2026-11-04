CXX      ?= g++
CXXFLAGS := -O3 -std=c++14 -DRAMULATOR -w -include cstdint
RAMDIR   := ramulator/src
OBJDIR   := build/obj
RAMSRCS  := $(filter-out $(RAMDIR)/Main.cpp $(RAMDIR)/Gem5Wrapper.cpp, $(wildcard $(RAMDIR)/*.cpp))
RAMOBJS  := $(patsubst $(RAMDIR)/%.cpp, $(OBJDIR)/%.o, $(RAMSRCS))

all: build/pimsim

$(OBJDIR)/%.o: $(RAMDIR)/%.cpp $(wildcard $(RAMDIR)/*.h)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/pimsim: src/pimsim.cpp $(RAMOBJS) $(wildcard $(RAMDIR)/*.h)
	$(CXX) $(CXXFLAGS) -I$(RAMDIR) -o $@ src/pimsim.cpp $(RAMOBJS)

clean:
	rm -rf build
.PHONY: all clean
