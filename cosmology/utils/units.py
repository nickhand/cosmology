"""
 units.py
 cosmology: unit conversions
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 09/02/2014
"""
import operator

# Note: operator here gives the function needed to go from 
# `absolute` to `relative` units
variables = {"wavenumber" : {'operator': operator.div, 'power' : 1}, \
             "distance" : {'operator': operator.mul, 'power' : 1}, \
             "volume" : {'operator': operator.mul, 'power' : 3}, \
             "power" : {'operator': operator.mul, 'power' : 3}, \
             "mass" : {'operator': operator.mul, 'power' : 1}, \
             "luminosity" : {'operator': operator.mul, 'power' : 1} }

def h_conversion_factor(variable_type, input_units, output_units, h):
    """
    Return the factor needed to convert between the units, dealing with 
    the pesky dimensionless Hubble factor, `h`.
    
    Parameters
    ----------
    variable_type : str
        The name of the variable type, must be one of the keys defined
        in ``units.variables``.
    input_units : str, {`absolute`, `relative`}
        The type of the units for the input variable
    output_units : str, {`absolute`, `relative`}
        The type of the units for the output variable
    h : float
        The dimensionless Hubble factor to use to convert
    """
    units_types = ['relative', 'absolute']
    if not all(t in units_types for t in [input_units, output_units]):
        raise ValueError("`input_units` and `output_units` must be one of %s" %units_types)
        
    if variable_type not in variables.keys():
        raise ValueError("`variable_type` must be one of %s" %variables.keys())
    
    if input_units == output_units:
        return 1.
    
    exponent = variables[variable_type]['power']
    if input_units == "absolute":
        return variables[variable_type]['operator'](1., h)**(exponent)
    else:
        return 1./(variables[variable_type]['operator'](1., h)**(exponent))
        
#end h_conversion_factor

#-------------------------------------------------------------------------------