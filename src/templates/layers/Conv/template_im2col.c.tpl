    // im2col
    for (i = 0; i < {{patches_height}}; ++i)
    {
        int i_offset = (i / {{kernel_w}}) % {{kernel_h}};
        int j_offset = i % {{kernel_w}};
        int c_offset = i / {{kernel_h}} / {{kernel_w}};
        
        for (h = 0; h < {{output_height}}; ++h)
        {
            for (w = 0; w < {{output_width}}; ++w)
            {
                int ii = h * {{strides}} - {{pad_top}} + i_offset;
                int jj = w * {{strides}} - {{pad_left}} + j_offset;

                int j = h*{{output_width}} + w;
                if (ii >= 0 && ii < {{input_height}} && jj >= 0 && jj < {{input_width}})
                    output_{{road}}[i*{{patches_width}} + j] = {{output_str}}[(c_offset*{{input_height}} + ii)*{{input_width}} + jj];
                else
                    output_{{road}}[i*{{patches_width}} + j] = 0;
            }
        }
    }