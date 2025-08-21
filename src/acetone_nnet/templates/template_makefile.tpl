CC = {{compiler}}

CFLAGS =
{{#compiler_flags}}
CFLAGS += {{.}}
{{/compiler_flags}}

LDFLAGS =
{{#linker_flags}}
LDFLAGS += {{.}}
{{/linker_flags}}

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

{{#bin_dataset}}
test_dataset.o: test_dataset.dat
	objcopy -I binary  -O {{.}} --add-symbol nn_test_inputs=.rodata:0 --rename-section .data=.rodata $< $@
{{/bin_dataset}}

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LBLIBS) $(LDFLAGS)

clean:
	rm $(EXEC) *.o