// synchronization flags
{{#flags}}
volatile int *synchro_{{src}}_{{dst}}=(int*) {{address}};
{{\flags}}

// communication tensors
{{#comm}}
volatile {{data_type}} *comm_{{src}}_{{dst}} __attribute__((aligned({{page_size}})));
{{/comm}}
