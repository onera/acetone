                new_f = ({{pads_front}} - f) % (2*{{channels_max}})
                if((new_f > {{channels_max}}))
                {
                    new_f = 2*{{channels_max}} - new_f;
                }

                new_i = ({{pads_top}} - i) % (2*{{height_max}})
                if((new_i > {{height_max}}))
                {
                    new_i = 2*{{height_max}} - new_i;
                }

                new_j = ({{pads_left}} - j) % (2*{{width_max}})
                if((new_j > {{width_max}}))
                {
                    new_j = 2*{{width_max}} - new_j;
                }