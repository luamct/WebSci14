// Copyright Jim Bosch 2011-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>
#include <iostream>
#include <math.h>

// Ugly but efficient
#define DEGREES(r) r*57.29577951308232
#define PIx2 6.283185307
#define MAX(a,b,c) (a>b ? (a>c ? a : c) : (b>c ? b : c))
#define MIN(a,b,c) (a<b ? (a<c ? a : c) : (b<c ? b : c))


namespace bp = boost::python;
namespace np = boost::numpy;


float degrees(float d)
{
	return d * 57.29577951308232; 
}

float hue(float r, float g, float b)
{
  // r*r + g*g + b*b - r*g - g*b - b*r
	float dem = sqrt(r*(r-g-b) + g*(g-b) + b*b);
	float H = (dem==0.0 ? 0.0 : acos((r-0.5*(g+b))/dem));
	return (b > g) ? PIx2-H : H;
}


/**
 * Conversion from RGB to IHLS color space.
 */
void rgb2ihls(float * img, float * ihls, int rows, int cols) 
{

	float r,g,b;
	for (int i=0; i<rows*cols*3; i+=3) 
	{		
		r = img[i];
		g = img[i+1];
		b = img[i+2];
		
		ihls[i] = hue(r, g, b);                       // H improved Hue
    ihls[i+1] = 0.2126*r + 0.7152*g + 0.0722*b;  	// Y Brightness
    ihls[i+2] = MAX(r,g,b) - MIN(r,g,b);          // S Saturation
                	
	}			
}

/**
 * Wrapper method for doing some safe checking and data adapting 
 * for the actual conversion method.
 */

void py_rgb2ihls(np::ndarray const & img, np::ndarray const & ihls) 
{
    if (img.get_dtype() != np::dtype::get_builtin<float>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type!");
        bp::throw_error_already_set();
    }
    if (img.get_nd() != 3) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions!");
        bp::throw_error_already_set();
    }
    if (ihls.get_dtype() != np::dtype::get_builtin<float>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        bp::throw_error_already_set();
    }
    if (ihls.get_nd() != 3) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        bp::throw_error_already_set();
    }

	int rows1 = img.shape(0);
	int rows2 = ihls.shape(0);
	int cols1 = img.shape(1);
	int cols2 = ihls.shape(1);
	
	if ((rows1 != rows2) && (cols1 != cols2)) {
		PyErr_SetString(PyExc_TypeError, "Incompatible images' dimensions.");
    bp::throw_error_already_set();
	}
	
    rgb2ihls(reinterpret_cast<float*>(img.get_data()),
    				 reinterpret_cast<float*>(ihls.get_data()), 
						 rows1, cols1);
}

BOOST_PYTHON_MODULE(_rgb2ihls) {
    np::initialize();  
    bp::def("rgb2ihls", py_rgb2ihls);
}
