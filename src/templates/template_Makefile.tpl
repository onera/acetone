CC = gcc
CFLAGS = -g -w -lm

SRC = {{source_files}}
HEADERS = {{header_files}}
OBJ = $(SRC:.cc=.o) $(HEADERS)
EXEC = {{function_name}}

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(LDFLAGS)  -o $@ $(OBJ) $(LBLIBS) $(CFLAGS)

clean:
	rm $(EXEC)