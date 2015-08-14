
ifeq (,$(filter _%,$(notdir $(CURDIR))))
  include target.mk
else
#----- End Boilerplate

# This section is for the architecture-specific target directory
VPATH =  $(SRCDIR)

# Additional headers and libraries directories
LIBLINEAR_PATH = ../../util/liblinear-2.01
JSGD_PATH = ../../util/jsgd-55/c

CC = g++ 
CXXFLAGS = -std=c++11 -Wall -I$(LIBLINEAR_PATH) -I$(JSGD_PATH)
LDLIBS = -L/usr/local/lib/ -L$(LIBLINEAR_PATH) -L$(JSGD_PATH)
LDFLAGS = -fopenmp

# Check if the configuration is Release or Debug
ifeq ($(CONFIGURATION),Debug)
	CXXFLAGS += -g
	LDLIBS += -g
else
	CXXFLAGS += -O3 -DNDEBUG#-funroll-loops
	LDLIBS += -O3
	CONFIGURATION = Release
endif

LDLIBS +=  -lboost_system -lboost_filesystem -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml \
-lopencv_imgcodecs -llinear -ljsgd /usr/lib64/libblas.so.3.4.2

ifeq ($(ATLAS),yes)
	CXXFLAGS += -D_USE_ATLAS_
	LDLIBS += -L/usr/lib64/atlas/
  	LDLIBS += -lcblas -latlas
endif


# Targets
PROGRAMS = test
all: $(PROGRAMS)
test: test.o LearnModels.o LMParams.o CharInstance.o Doc.o PedroFeatures.o LibLinearWrapper.o JsgdWrapper.o
LearnModels.o : LearnModels.cpp LearnModels.h HogSvmModel.h
CharInstance.o : CharInstance.cpp CharInstance.h commonTypes.h
LMParams.o : LMParams.cpp LMParams.h commonTypes.h
Doc.o : Doc.cpp Doc.h commonTypes.h
PedroFeatures.o : PedroFeatures.cpp PedroFeatures.h commonTypes.h



#----- Begin Boilerplate
endif
