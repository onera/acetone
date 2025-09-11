                if(f >= {{channels_and_pad_front}})
                {
                    new_f = (new_f) % ({{input_channels}});
                }
                if(f < {{pads_front}})
                {
                    new_f = (new_f) % ({{input_channels}}) +  {{input_channels}};
                }

                if(i >= {{height_and_pad_top}})
                {
                    new_i = (new_i) % ({{input_height}});
                }
                if(i < {{pads_top}})
                {
                    new_i = (new_i) % ({{input_height}}) + {{input_height}};
                }

                if(j >= {{width_and_pad_left}})
                {
                    new_j = (new_j) % ({{input_width}});
                }
                if(j < {{pads_left}})
                {
                    new_j = (new_j) % ({{input_width}}) + {{input_width}};
                }