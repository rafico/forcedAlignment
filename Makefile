forcedAlignment := forcedAlignmentCPP
jsgd := util/jsgd-55/c
liblinear := util/liblinear-2.01
directories :=  $(liblinear) $(jsgd) $(forcedAlignment)

all : $(directories)

.PHONY: $(directories)

$(directories):
	$(MAKE) --directory=$@