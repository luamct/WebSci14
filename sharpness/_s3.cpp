
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <math.h>
#include <fftw3.h>

#define PR(v,s) for (int i=0;i<s;i++) std::cout<<v[i]<<" ";std::cout<<endl;
#define PIx2 6.28318530717958


namespace bp = boost::python;


class S3
{
private:

	// Configuration and static values
	int s1_block_size;;
	int s2_block_size;
	int step;

	double* w;

	// Constants
	int k1, k2, t1, t2;

public:

	S3(int k1, int k2, int t1, int t2)
	{
		this->s1_block_size = 32;
		this->s2_block_size = 8;
		this->step = 8;

		this->k1 = k1;
		this->k2 = k2;
		this->t1 = t1;
		this->t2 = t2;

		hanning(s1_block_size);
	}

	~S3() {
		delete[] w;
	}

	/**
	 * Creates the 2D Hanning window for smoothing the blocks and avoiding edges problems.
	 */
	void hanning(int n) 
	{
		// Hanning 1D filter
		double* han = new double[n];
		for (int i=0; i<n; ++i) {
			han[i] = 0.5*(1-cos(PIx2*(i+1)/(n+1)));
		}

		// Allocate window and set it as the matrix product of 
		// the hanning 1D filter and its transpose
		this->w = new double[n*n];
    for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
        this->w[i*n + j] = han[i] * han[j];
      }
    }

		delete[] han;
	}

	void copy_block(uint8_t* img, double* blk, int n, int r, int c, int rows, int cols) 
	{
		int hb = n/2;
		r = r-hb;
		c = c-hb;

		// Hold the fixed indexes simulating a padding of the image. The pad is half a block_size 
		// on each of the borders as a mirror of the image's original border pixels.
		int pr, pc;
    for (int br=0; br<n; br++) {
      for (int bc=0; bc<n; bc++) {	
				pr = r+br;
				pc = c+bc;
				pr = (pr<0 ? abs(pr)-1 : pr);
				pc = (pc<0 ? abs(pc)-1 : pc);
				pr = (pr >= rows ? 2*(rows-1) - pr + 1 : pr);
				pc = (pc >= cols ? 2*(cols-1) - pc + 1 : pc);

				blk[br*n + bc] = img[pr*cols + pc];
			}
		}
	}

	/**
	 * Calculates the discrete fourier transform on the image block.
	 */
	void fft(double* blk, double* out, int n) 
	{
		int nh = n/2 + 1;
	  fftw_complex* complex_out = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * n * nh );
		
	  fftw_plan plan = fftw_plan_dft_r2c_2d ( n, n, blk, complex_out, FFTW_ESTIMATE );
	  fftw_execute ( plan );
	
		// Get the absolute value of each value
		for (int i=0; i < n; i++ ) {
			for (int j=0; j < nh; j++ ) {
				out[i*n + j] = sqrt(pow(complex_out[i*nh + j][0],2) + pow(complex_out[i*nh + j][1], 2)) ;
			}
		}

		// FFTW returns an incomplete matrix due to the FFT's symmetry,
		// so here we fill the rest for easier manipulation
		int mirror = 2*(nh-1);
		for (int i=0; i < n; i++) {
			for (int j=nh; j < n; j++) {
				out[i*n + j] = out[(i==0 ? 0 : n-i)*n + mirror-j];
			}
		}

		// Deallocation
		fftw_destroy_plan(plan);
		fftw_free(complex_out);
	}


	void spectrum_magn(double* dft, double* spec, int n)
	{
		double ex, ey, f11, f12, f21, f22;
		double th, x, y;
		int x1, x2, y1, y2;

		double rad = M_PI/180;
		int dr = 360;

		// For each polar frequency
		for (int r=1; r<=n/2; ++r) 
		{
	    double zs = 0.0;
 
 			// For each 1 degree angle
  	  for (int ith=0; ith<dr; ++ith) 
			{
        th = ith*rad;
        x = r * sin(th);
        y = r * cos(th);

				// Get the nearest indexes for bilinear interpolation
        x1 = int(floor(x));
        x2 = int(ceil (x));
				y1 = int(floor(y));
        y2 = int(ceil (y));

        ex = (x>=0 ? fabs(x-x1) : fabs(x-x2));
        ey = (y>=0 ? fabs(y-y1) : fabs(y-y2));

        if (x1<0) {
					x1 += n;
					if (x2<0) 
						x2 += n;
				}

        if (y1<0) {
					y1 += n;
					if (y2<0) 
						y2 += n;
				}

				// Get boundaries values and interpolate bilinearly
        f11 = dft[x1*n + y1];
        f12 = dft[x1*n + y2];
        f21 = dft[x2*n + y1];
        f22 = dft[x2*n + y2];
    
				zs += f11 + (f12-f11)*(1-ex)*ey + (f21-f11)*(1-ey)*ex + (f22-f11)*ex*ey;
			}

			spec[r-1] = zs/dr;
		}
	}

	double linefit(double* x, double *y, int n) 
	{
		double sx=0, sy=0, ssx=0, sxy=0;
		for (int i=0; i<n; ++i) {
			sx  += x[i];
			sy  += y[i];
			sxy += x[i]*y[i];
			ssx += x[i]*x[i];
		}
		return (sxy - sx*sy/n)/(ssx - sx*sx/n);
	}


	double sigmoid(double v) {
		return 1.0 - 1.0/(1.0 + exp(k1 * (k2 - v)));
	}


	void print(double* blk, int n) {
		for (int r=0; r < n; r++) {
			for (int c=0; c < n; c++) {
				std::cout << blk[r*n + c] << " ";
			}
			std::cout << std::endl;
		}		
	}

	/**
	 * Check if there is enough contrast on the given block;
	 */
	bool enough_contrast(double* blk, int n) 
	{	
		double l, lmean=0.0, lmax=0.0, lmin=255.0;

		for (int p=0; p<n*n; ++p) {
			l = pow( 0.7656 + 0.0364*blk[p], 2.2 );
			lmean += l;
			if (l > lmax) lmax = l;
			if (l < lmin) lmin = l;
		}
		lmean /= n*n;

		return (lmax-lmin > t1) && (lmean > t2);
	}

	/**
	 * Set value on a block of the output map.
	 */
	void set_map(double* map, double value, int n, int r, int c, int rows, int cols)
	{
		int hn = n/2;
		for (int mr=r-hn; mr < r+hn; ++mr) {
			for (int mc=c-hn; mc < c+hn; ++mc) 
			{
				// Only set if coords are inside the image
				if ( (mr>=0) && (mr<rows) && 
				     (mc>=0) && (mc<cols) ) 
				{
					map[mr*cols + mc] = value;
				}
			}
		}
	}

	/**
	 * Calculates the S1 map based on the image's frequencies spectrum.
	 */
	void s1(uint8_t* img, double* map, int rows, int cols) 
	{
		int n     = s1_block_size;  // Alias
		int pad   = n/2;            // Pad size on the borders
		double fd = 1.0/n;  		    // Distance between frequencies on the spectrum
		double alpha;               // Linear regression slope

		// Allocate reusable data
		double* blk  = new double[n*n];  // Image block
		double* dft  = new double[n*n];  // Fourier transformed block
		double* spec = new double[n/2];  // Spectrum magnitude summed over orientations
		double* freq = new double[n/2];  // Frequencies associated with the magnitudes
		
		for (int r=0; r < (rows + pad); r+=step) {
			for (int c=0; c < (cols + pad); c+=step) {

				// Copy the block centered on (r,c) to blk including a 
				// mirror padding on the borders
				copy_block(img, blk, n, r, c, rows, cols);

				if (enough_contrast(blk, n)) 
				{				
					// Apply hanning window for smoothing
					for (int i=0; i<n*n; i++) {
						blk[i] *= this->w[i];
					}

					fft(blk, dft, n);
					spectrum_magn(dft, spec, n);

					// Convert to log
					for (int i=0; i<n/2; i++) {
						freq[i] = log((i+1)*fd);
						spec[i] = log(spec[i]);
					}

					// Linear regression to find the slope of the spectrum followed by 
					// a sigmoid to tune the value to the human's visual system
					alpha = sigmoid( -linefit(freq, spec, n/2) );
				}
				else {
					// Not enough contrast. Just set value to zero.
					alpha = 0.0;
				}

				// Set value into block on output map
				set_map(map, alpha, step, r, c, rows, cols);
			}
		}

		delete[] blk;
		delete[] dft;
		delete[] spec;
		delete[] freq;
	}


	double max_variation(double* blk, int n) 
	{
		double var, vmax = 0.0;
		for (int r=0; r<n-1; ++r) {
			for (int c=0; c<n-1; ++c) {

				// Sum variations of all possible neighbors on a 2x2 block
				var = ( fabs( blk[r*n + c]     - blk[(r+1)*n + c]     ) + // Horizontal neighbors 
							  fabs( blk[r*n + (c+1)] - blk[(r+1)*n + (c+1)] ) + 
							  fabs( blk[r*n + c]     - blk[r*n + (c+1)]     ) + // Vertical neighbors
							  fabs( blk[(r+1)*n + c] - blk[(r+1)*n + (c+1)] ) + 
							  fabs( blk[r*n + c]     - blk[(r+1)*n + (c+1)] ) + // Diagonal
							  fabs( blk[(r+1)*n + c] - blk[r*n + (c+1)]     ) );

				if (var > vmax) {
					vmax = var;
				}
			}
		}
		return vmax/255.0;
	}

	/**
	 * Calculates the S2 map based on spatial contrast of the image.
	 */
	void s2(uint8_t* img, double* map, int rows, int cols) 
	{
		int n   = s2_block_size;  // Alias
		int pad = n/2;            // Pad size on the borders
		double var;               // Variation on each block

		double* blk  = new double[n*n];  // Image block

		for (int r=0; r < (rows + pad); r+=n) {
			for (int c=0; c < (cols + pad); c+=n) {

				// Copy the block centered on (r,c) to blk including a 
				// mirror padding on the borders
				copy_block(img, blk, n, r, c, rows, cols);

				// Get the maximum variation across all 2x2 blocks and normalize
				var = max_variation(blk, n)/4;

				set_map(map, var, n, r, c, rows, cols);
			}
		}

		delete[] blk;
	}

	
	void s3(uint8_t* img, double* map, int rows, int cols) 
	{
		// allocate output maps
		double* s1_map = new double[rows*cols];
		double* s2_map = new double[rows*cols];

		s1(img, s1_map, rows, cols);
		s2(img, s2_map, rows, cols);


		for (int p=0; p<rows*cols; ++p) {
			map[p] = sqrt(s1_map[p]) * sqrt(s2_map[p]);
		}

		delete[] s1_map;
		delete[] s2_map;
	}

	void py_s3(pyublas::numpy_vector<uint8_t> _img, pyublas::numpy_vector<double> _map)
	{
		uint8_t* img   = _img.data().data();
		double* s3_map = _map.data().data(); 

		// Get image dimensions
		int rows = (int)_img.dims()[0];
		int cols = (int)_img.dims()[1];

		s3(img, s3_map, rows, cols);
	}
	
/*
	void py_test(pyublas::numpy_vector<double> mat, pyublas::numpy_vector<double> out) 
	{
		int n = mat.dims()[0];
		double* c_mat = mat.data().data();
		double* c_out = out.data().data();

		cout << linefit(c_mat, c_out, n) << endl;

		//fft(c_mat, c_out, n);
		//cout << c_mat[0] << " " << c_mat[1] << endl;	
		//spectrum_magn(c_mat, c_out, n);

	}
	*/
};

/*
int main()
{
	int rows, cols;
	FILE* f = fopen("orchids.txt", "r");
	fscanf(f, "%d %d", &rows, &cols);

	printf("%d %d\n", rows, cols);
	uint8_t* img = new uint8_t[rows*cols];
	
	int v;
	for (int p=0; p<rows*cols; ++p) {
		fscanf(f, "%d", &v);
		img[p] = (uint8_t)v;
	}

	//printf("%d\n", img[4]);
	//printf("%d\n", img[1*cols + 1]);
	
	double* map = new double[rows*cols];

	S3 s3;
	s3.s3(img, map, rows, cols);

	
	printf("%f\n", map[0]);
	
	delete[] img;
	delete[] map;

	fclose(f);
}
*/


BOOST_PYTHON_MODULE(_s3)
{
	bp::class_<S3>("S3", bp::init<int, int, int, int>())
//  	.def("test", &S3::py_test)
  	.def("s3", &S3::py_s3);
}

