CC = gcc
CFLAGS = -g -w -lm

SRC = inference.c global_vars.c main.c test_dataset.c
HEADERS = inference.h test_dataset.h
OBJ = $(SRC:.cc=.o) $(HEADERS)
EXEC = inference

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(LDFLAGS)  -o $@ $(OBJ) $(LBLIBS) $(CFLAGS)

clean:
	rm $(EXEC)