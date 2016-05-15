
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
-lopencv_imgcodecs -llinear -ljsgd -lblas

ifeq ($(ATLAS),yes)
	CXXFLAGS += -D_USE_ATLAS_
	LDLIBS += -L/usr/lib64/atlas/
  	LDLIBS += -lcblas -latlas
endif


# Targets
PROGRAMS = test
all: $(PROGRAMS)
test: test.o CharClassifier.o Params.o Doc.o HogUtils.o LibLinearWrapper.o JsgdWrapper.o TrainingData.o HogSvmModel.o ForcedAlignment.o Classifier.o CharSequence.o Dataset.o TranscriptLexicon.o AnnotatedLine.o
CharClassifier.o : CharClassifier.cpp CharClassifier.h HogSvmModel.h
Params.o : Params.cpp Params.h commonTypes.h
Doc.o : Doc.cpp Doc.h commonTypes.h
HogUtils.o : HogUtils.cpp HogUtils.h commonTypes.h
CharSequence.o : CharSequence.cpp CharSequence.h commonTypes.h
AnnotatedLine.o : AnnotatedLine.cpp AnnotatedLine.h commonTypes.h
TrainingData.o : TrainingData.cpp TrainingData.h commonTypes.h
HogSvmModel.o : HogSvmModel.cpp HogSvmModel.h commonTypes.h
ForcedAlignment.o : ForcedAlignment.cpp ForcedAlignment.h commonTypes.h
Classifier.o : Classifier.cpp Classifier.h commonTypes.h
TranscriptLexicon.o : TranscriptLexicon.cpp TranscriptLexicon.h commonTypes.h CharSequence.cpp Params.cpp
Dataset.o : Dataset.cpp Dataset.h commonTypes.h CharSequence.cpp


#----- Begin Boilerplate
endif
