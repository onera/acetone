                if((f < {{pads_front}}) || (f >= {{channels_and_pad_front}}))
                {
                    new_f = (new_f) % ({{input_channels}});
                }

                if((i < {{pads_top}}) || (i >= {{height_and_pad_top}}))
                {
                    new_i = (new_i) % ({{input_height}});
                }

                if((j < {{pads_left}}) || (j >= {{width_and_pad_left}}))
                {
                    new_j = (new_j) % ({{input_width}});
                }