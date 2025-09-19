// synchronization flags
{{#flags}}
extern volatile int *synchro_{{src}}_{{dst}};
{{/flags}}

// communication tensors
{{#comms}}
extern volatile {{data_type}} *comm_{{src}}_{{dst}} __attribute__((aligned({{page_size}})));
{{/comms}}
