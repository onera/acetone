"""
 *******************************************************************************
 * ACETONE: Predictable programming framework for ML applications in safety-critical systems
 * Copyright (c) 2022. ONERA
 * This file is part of ACETONE
 *
 * ACETONE is free software ;
 * you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation ;
 * either version 3 of  the License, or (at your option) any later version.
 *
 * ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this program ;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
 ******************************************************************************
"""

import code_generator.layers.Conv_layers.Conv2D as Conv2D

class Conv2D_6loops(Conv2D.Conv2D):
    """Implements Conv2D using the six-loops algorithm (direc conv)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.nb_filters) + '; ++f)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')
        source_file.write('                sum = 0;\n')
        source_file.write('                for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n                {\n')
        source_file.write('                    for (int m = 0; m < '+str(self.kernel_h)+'; ++m)\n                    {\n')
        source_file.write('                        for (int n = 0; n < '+str(self.kernel_w)+'; ++n)\n                        {\n')
        source_file.write('                            int ii = i*'+str(self.strides)+' + m*'+str(self.dilation_rate)+' - '+str(self.pad_left)+';\n')
        source_file.write('                            int jj = j*'+str(self.strides)+' + n*'+str(self.dilation_rate)+' - '+str(self.pad_top)+';\n\n')
        source_file.write('                            if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                            {\n')
        
        source_file.write('                                sum += '+output_str+'[jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[n + '+str(self.kernel_w)+'*(m + '+str(self.kernel_h)+'*(c + '+str(self.input_channels)+'*f))];\n')
        source_file.write('                            }\n                        }\n                    }\n                }\n')                                            
        source_file.write('                sum += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[f];\n'            )
        
        a = self.activation_function.write_activation_str(self.local_var)
        
        if(self.fused_layer):
            b=self.fused_layer.write_activation_str(self.local_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)')
            if(self.activation_function.name != 'linear'):
                source_file.write('                '+self.local_var+' = '+a+';\n')
            source_file.write('                output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)] = '+ b +';\n')
        else:
            source_file.write('                output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)] = '+ a +';\n')
        source_file.write('            }\n        }\n    }\n\n')
