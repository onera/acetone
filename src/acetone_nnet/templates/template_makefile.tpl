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

all: $(EXEC) $(EXEC).so

{{#bin_dataset}}
test_dataset.o: test_dataset.dat
	objcopy -I binary  -O {{.}} --add-symbol nn_test_inputs=.rodata:0 --rename-section .data=.rodata $< $@

parameters.o: parameters.dat
	objcopy -I binary  -O {{.}} {{symtab}} --set-section-alignment .data={{align}} --rename-section .data=.rodata $< $@

{{/bin_dataset}}


$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LBLIBS) $(LDFLAGS)

$(EXEC).so: inference.c parameters.o train_hook.c global_vars.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $^ $(LBLIBS) $(LDFLAGS)

clean:
	rm -f $(EXEC) *.o