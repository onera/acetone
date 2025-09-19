{{#main_core}}
{{#flags}}
    *synchro_{{src}}_{{dst}}=0;
{{/flags}}
{{/main_core}}
{{^main_core}}
    sleep(1);
{{/main_core}}