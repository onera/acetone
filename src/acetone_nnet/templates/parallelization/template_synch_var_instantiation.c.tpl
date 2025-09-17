// synchronization flags
{{#flags}}
volatile int *synchro_{{src}}_{{dst}} = (volatile int*) {{address}};
{{\flags}}

// communication tensors
{{#comm}}
volatile {{data_type}} *comm_{{src}}_{{dst}} __attribute__((aligned({{page_size}}))) = (volatile {{data_type}}*) {{address}};
{{/comm}}
