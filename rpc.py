import numpy as np
from osgeo import gdal

def apply_poly(poly, x, y, z):
        """
        Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.
        将三次多项式的统一模式构建为一个单独的函数
        Args:
            poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
                ordered following the RPC convention.
            x, y, z: triplet of floats. They may be numpy arrays of same length.

        Returns:
            the value(s) of the polynom on the input point(s).
        """
        out = 0
        out += poly[0]
        out += poly[1]*y + poly[2]*x + poly[3]*z
        out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
        out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
        out += poly[10]*x*y*z
        out += poly[11]*y*y*y
        out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
        out += poly[15]*x*x*x
        out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
        out += poly[19]*z*z*z
        return out

def apply_rfm(num, den, x, y, z):
        """
        Evaluates a Rational Function Model (rfm), on a triplet of numbers.
        执行20个参数的分子和20个参数的除法
        Args:
            num: list of the 20 coefficients of the numerator
            den: list of the 20 coefficients of the denominator
                All these coefficients are ordered following the RPC convention.
            x, y, z: triplet of floats. They may be numpy arrays of same length.

        Returns:
            the value(s) of the rfm on the input point(s).
        """
        return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)

def obj2photo(img_info,lon, lat, alt):
        """
        Convert geographic coordinates of 3D points into image coordinates.
        正投影：从地理坐标到图像坐标
        Args:
            lon (float or list): longitude(s) of the input 3D point(s)
            lat (float or list): latitude(s) of the input 3D point(s)
            alt (float or list): altitude(s) of the input 3D point(s)
 
        Returns:
            float or list: horizontal image coordinate(s) (column index, ie x)
            float or list: vertical image coordinate(s) (row index, ie y)
        """
        nlon = (np.asarray(lon) - img_info['lon_off']) / img_info['lon_scale']
        nlat = (np.asarray(lat) - img_info['lat_off']) / img_info['lat_scale']
        nalt = (np.asarray(alt) - img_info['hei_off']) / img_info['hei_scale']
 
        col = apply_rfm(img_info['SNUM'], img_info['SDEN'], nlat, nlon, nalt)
        row = apply_rfm(img_info['LNUM'], img_info['LDEN'], nlat, nlon, nalt)
 
        col = col * img_info['sample_scale'] + img_info['sample_off']
        row = row * img_info['line_scale'] + img_info['line_off']
 
        return col, row
 
 
def photo2obj(img_info,col, row, alt, return_normalized=False):
        """
        Convert image coordinates plus altitude into geographic coordinates.
        反投影：从图像坐标到地理坐标
        Args:
            col (float or list): x image coordinate(s) of the input point(s)
            row (float or list): y image coordinate(s) of the input point(s)
            alt (float or list): altitude(s) of the input point(s)
 
        Returns:
            float or list: longitude(s)
            float or list: latitude(s)
        """
        ncol = (np.asarray(col) - img_info['sample_off']) / img_info['sample_scale']
        nrow = (np.asarray(row) - img_info['line_off']) / img_info['line_scale']
        nalt = (np.asarray(alt) - img_info['hei_off']) / img_info['hei_scale']
 
        lon = apply_rfm(img_info['LONNUM'], img_info['LONDEN'], nrow, ncol, nalt)
        lat = apply_rfm(img_info['LATNUM'], img_info['LATDEN'], nrow, ncol, nalt)

        if not return_normalized:
            lon = lon * img_info['lon_scale'] + img_info['lon_off']
            lat = lat * img_info['lat_scale'] + img_info['lat_off']
 
        return lon, lat
 
 
def photo2obj_iter(img_info,col, row, alt):
        """
        Iterative estimation of the localization function (image to ground),
        for a list of image points expressed in image coordinates.
        逆投影时的迭代函数
        Args:
            col, row: normalized image coordinates (between -1 and 1)
            alt: normalized altitude (between -1 and 1) of the corresponding 3D
                point
 
        Returns:
            lon, lat: normalized longitude and latitude
 
        Raises:
            MaxLocalizationIterationsError: if the while loop exceeds the max
                number of iterations, which is set to 100.
        """
        # target point: Xf (f for final)
        Xf = np.vstack([col, row]).T
 
        # use 3 corners of the lon, lat domain and project them into the image
        # to get the first estimation of (lon, lat)
        # EPS is 2 for the first iteration, then 0.1.
        lon = -col ** 0  # vector of ones
        lat = -col ** 0
        EPS = 2
        x0 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat, lon, alt)
        y0 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat, lon, alt)
        x1 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat, lon + EPS, alt)
        y1 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat, lon + EPS, alt)
        x2 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat + EPS, lon, alt)
        y2 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat + EPS, lon, alt)
 
        n = 0
        while not np.all((x0 - col) ** 2 + (y0 - row) ** 2 < 1e-18):
            X0 = np.vstack([x0, y0]).T
            X1 = np.vstack([x1, y1]).T
            X2 = np.vstack([x2, y2]).T
            e1 = X1 - X0
            e2 = X2 - X0
            u  = Xf - X0
 
            # project u on the base (e1, e2): u = a1*e1 + a2*e2
            # the exact computation is given by:
            #   M = np.vstack((e1, e2)).T
            #   a = np.dot(np.linalg.inv(M), u)
            # but I don't know how to vectorize this.
            # Assuming that e1 and e2 are orthogonal, a1 is given by
            # <u, e1> / <e1, e1>
            num = np.sum(np.multiply(u, e1), axis=1)
            den = np.sum(np.multiply(e1, e1), axis=1)
            a1 = np.divide(num, den).squeeze()
 
            num = np.sum(np.multiply(u, e2), axis=1)
            den = np.sum(np.multiply(e2, e2), axis=1)
            a2 = np.divide(num, den).squeeze()
 
            # use the coefficients a1, a2 to compute an approximation of the
            # point on the gound which in turn will give us the new X0
            lon += a1 * EPS
            lat += a2 * EPS
 
            # update X0, X1 and X2
            EPS = .1
            x0 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat, lon, alt)
            y0 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat, lon, alt)
            x1 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat, lon + EPS, alt)
            y1 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat, lon + EPS, alt)
            x2 = apply_rfm(img_info['SNUM'], img_info['SDEN'], lat + EPS, lon, alt)
            y2 = apply_rfm(img_info['LNUM'], img_info['LDEN'], lat + EPS, lon, alt)
    
            n += 1
 
        return lon, lat
