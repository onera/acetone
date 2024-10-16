CC = {{compiler}}

CFLAGS =
{{#compiler_flags}}
CFLAGS += {{.}}
{{/compiler_flags}}

SRC =
{{#source_files}}
SRC += {{.}}
{{/source_files}}

HEADERS =
{{#header_files}}
HEADERS += {{.}}
{{/header_files}}

OBJ = $(SRC:.c=.o) $(HEADERS)
EXEC = {{executable_name}}

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(LDFLAGS)  -o $@ $(OBJ) $(LBLIBS) $(CFLAGS)

clean:
	rm $(EXEC) *.o