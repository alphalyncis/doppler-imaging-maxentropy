"""
# Original author: Ian Crossfield (Python 2.7)

Planetary mapping routines.

phi = 0 faces toward the observer
phi = pi thus faces away from the observer
theta=pi/2 is the z-axis or 'north pole' 
theta=-pi/2 is the 'south pole' -- this is in fact not true, theta=(0, pi)
"""

######################################################
# 23-03-2023 Xueqing Chen: added equal area grids
# 18-02-2020 Emma Bubb: change to run on Python 3

######################################################
# 2010-01-15 20:31 IJC: Started                   .               .            .
# 2013-08-07 11:04 IJMC: Added mu field for cells and maps

from numpy import pi
import numpy as np
import matplotlib.pyplot as plt

def polyarea(x, y):
    """Compute the area of a polygon whose vertices are at the points (x,y).

    :INPUTS:
      x, y : 1D sequences
        Cartesian coordinates of the (non-intersecting) polygon.

    :REFERENCE:
      http://mathworld.wolfram.com/PolygonArea.html
    """
    # 2013-05-29 12:18 IJMC: Created

    area = 0.
    npts = max(len(x), len(y))
    for ii in range(npts): # EB: xrange to range
        area += x[ii]*y[(ii+1) % npts] - x[(ii+1) % npts]*y[ii]
    return  np.abs(area*0.5)

def make_latlon_grid(nphi, ntheta):
    """Make grids of phi and theta values with the specified number of
    points in each direction.  Phi ranges from 0 to 2pi, and theta
    ranges from -pi/2 (in fact 0) to pi/2 (in fact pi).

    Returns meshgird(phi, theta)
    """
    # 2010-01-15 20:29 IJC: Created
    # 2013-08-18 15:57 IJMC: Updated so phi values don't repeat at 0 & 2pi
    # 2023-03-25 XQ: make equal area grids

    phi, theta = np.meshgrid(np.linspace(0,2*pi,nphi+1), np.linspace(0, pi,ntheta+1))

    return phi, theta

def make_eqarea_grid(ncell, verbose=False):
    """Make grids of phi and theta values with the specified number of
    cells of roughly equal area. Phi ranges from 0 to 2pi, and theta
    ranges from -pi/2 (in fact 0) to pi/2 (in fact pi).

    Returns:
        phi: List of 1d arrays of size N_cells_per_row
        theta: 1d array of size nlat (number of rows)
    """
    # 2023-03-25 XQ: make equal area grids
    def find_number_of_rows(Ncell, m0=10):
        diff_old = 1e10
        m = m0
        while True:
            if Ncell < 5:
                raise ValueError("Number of cells is too small. At least 5 cells are needed.")
            ncells_per_row = np.array([int(2 * m * np.cos(n*np.pi/m)) for n in range(1, int(m/2))])
            Ncell_new = 2 * np.sum(ncells_per_row)
            diff = Ncell - Ncell_new
            if np.abs(diff) > np.abs(diff_old): # right amount of cells, return the previous one
                return 2*len(ncells_per_row_old), Ncell_old, ncells_per_row_old 
            if diff > 0: # need more cells
                m += 2
            else: # need less cells
                m -= 2
            Ncell_old = Ncell_new
            ncells_per_row_old = ncells_per_row
            diff_old = diff

    nlat, ncell_true, ncells_per_row = find_number_of_rows(ncell)
    ncells_per_row = np.concatenate([np.flip(ncells_per_row), ncells_per_row])
    height = np.pi / nlat
    theta = np.array([height/2 + m * height for m in range(0, nlat)])
    widths = np.pi * 2 / ncells_per_row
    phi = [None for m in range(nlat)]
    for m in range(nlat):
        phi[m] = np.array([widths[m]/2 + n * widths[m] for n in range(ncells_per_row[m])])
    
    if verbose:
        print(f"Created equa-area grid of {ncell_true} cells, in {len(theta)} latitude grids with {ncells_per_row} lontitude cells.")

    return phi, theta, height, widths, ncell_true

def makespot(spotlat, spotlon, spotrad, phi, theta):
    """
    :INPUTS:
      spotlat : scalar
        Latitude of spot center, in degrees, from 0 to 180 (actually from -90 to 90)

      spotlon : scalar
        Longitude of spot center, in degrees, from 0 to 360

      spotrad : scalar
        Radius of spot, in radians. degrees

      phi, theta : 2D NumPy arrays
         output from :func:`makegrid`.  Theta ranges from -pi/2 to +pi/2.

    :EXAMPLE:
      ::

        import maps
        nlat, nlon = 60, 30
        phi, theta = maps.makegrid(nlat, nlon)
        # Make a small spot centered near, but not at, the equator:
        equator_spot = maps.makespot(0, 0, 23, phi, theta)
        # Make a larger spot centered near, but not at, the pole:
        pole_spot = maps.makespot(68, 0, 40, phi, theta)

      ::

        import maps
        nlat, nlon = 60, 30
        map = maps.map(nlat, nlon, i=0., deltaphi=0.)
        phi = map.corners_latlon.mean(2)[:,1].reshape(nlon, nlat)
        theta = map.corners_latlon.mean(2)[:,0].reshape(nlon, nlat) - np.pi/2.
        # Make a small spot centered near, but not at, the equator:
        equator_spot = maps.makespot(0, 0, 23, phi, theta)
        # Make a larger spot centered near, but not at, the pole:
        pole_spot = maps.makespot(68, 0, 40, phi, theta)

    """
    # 2013-08-18 16:01 IJMC: Created

    spotlat *= (np.pi/180)
    spotlon *= (np.pi/180)
    spotrad *= (np.pi/180)

    pi2 = 0.5*np.pi
    xyz = np.array((np.cos(phi) * np.sin(theta + pi2), np.sin(phi) * np.sin(theta + pi2), np.cos(theta + pi2))).reshape(3, phi.size)

    # First rotate around z axis, to align spot with sub-observer meridian
    # Then, rotate around y axis, to align spot with pole.
    zrot = np.array([[np.cos(np.pi-spotlon), -np.sin(np.pi-spotlon), 0], [np.sin(np.pi-spotlon), np.cos(np.pi-spotlon), 0.], [0,0,1]])
    yrot = np.array([[np.cos(spotlat+pi2), 0, np.sin(spotlat+pi2)], [0,1,0], [-np.sin(spotlat+pi2), 0, np.cos(spotlat+pi2)]])
    xyz = np.dot(np.dot(yrot, zrot), xyz)

    # Convert Cartesian to spherical coordinates
    ang = np.arccos(xyz[2])

    # Spot is where (theta - theta_pole) < radius.
    spotmap = ang.T <= spotrad

    return spotmap.reshape(phi.shape)

def profile_spotmap(param, *args, **kw):
    """Model line profiles, assuming a simple one-spot model.

    phi, theta, R = args[0:3]
    startemp, spottemp, spotlat, spotlon, spotrad = param[0:5]
    OR
    startemp, temp1, lat1, lon1, rad1, temp2, lat2, lon2, rad2 = param[0:9]
    """
    # 2013-08-19 09:59 IJMC: Created
    # 2013-08-27 10:45 IJMC: Updated to multi-spot-capable
    phi, theta, R = args[0:3]
    nparam = len(param)
    nspots = int((nparam-1)/4)
    startemp = param[0]
    map_pixels = np.ones(phi.shape) * param[0]
    for ii in range(nspots):
        spottemp, spotlat, spotlon, spotrad = param[1+ii*4:1+(ii+1)*4]
        boolspot = makespot(spotlat, spotlon, spotrad, phi, theta).astype(np.float32)
        map_pixels -= boolspot * (startemp - spottemp)

    return np.dot(map_pixels.ravel(), R)


class MapCell:
    def __init__(self):
        self.corners = np.zeros((3, 4), dtype=float)
        self.corners_latlon = np.zeros((2, 4), dtype=float)
        self.vcorners = np.zeros((3, 4), dtype=float)
        self.rvcorners = np.zeros(4, dtype=float)
        self.visible_corners = np.zeros((3, 4), dtype=float)
        self.visible_vcorners = np.zeros((3, 4), dtype=float)
        self.visible_rvcorners = np.zeros(4, dtype=float)
        self.projected_area = 0.
        self.mu = 0.
        return

    def get_mu(self):
        ### Compute mu:
        normal_vector = np.dot(np.linalg.pinv(self.corners.T), np.ones(4))
        self.mu = normal_vector[0] / np.sqrt(np.dot(normal_vector, normal_vector))
        return

    def get_projected_area(self, inc):
        if (self.corners[0] <= 0).all():
            # cell is hidden, on the back side.
            area = 0. 
            self.visible_corners = self.corners * np.nan
        elif (self.corners[0] > 0).all():
            # cell is completely visible, on the front side.
            self.visible_corners = self.corners
            y = self.corners[1]
            z = self.corners[2]
            
            inds = np.argsort(np.arctan2(z-z.mean(), y-y.mean()))
            area = polyarea(y[inds], z[inds])
        else:
            # Cell is only partially visible (on the limb). Find the
            # nearest point on on the limb, with the same latitude as
            # each vertex.
            visible_corners = self.corners.copy()
            back_indices = (visible_corners[0] < 0).nonzero()[0]
            for ii in back_indices:
                newx = 0. # on the limb!
                newy = np.sin(self.corners_latlon[0,ii]) * \
                    np.sqrt(1. - np.tan(inc)**2 / np.tan(self.corners_latlon[0,ii])**2)
                if visible_corners[1,ii]/newy < 0: 
                    newy *= -1
                newz = np.cos(self.corners_latlon[0,ii]) / np.cos(inc)
                visible_corners[:, ii] = newx, newy, newz

            if not (np.isfinite(visible_corners)).all():
                self.visible_corners = self.corners * np.nan
                area = 0
                #print("Non-finite projected corners; need to fix this.") # EB updated print statement
            else:
                self.visible_corners = visible_corners

                y = self.visible_corners[1]
                z = self.visible_corners[2]
                #yz = np.array(zip(y,z)) #2017-01-10 13:04 IJMC: removed: np.unique(zip(y,z))
                #inds = np.argsort(np.arctan2(yz[:,1]-yz[:,1].mean(), yz[:,0]-yz[:,0].mean()))
                #area = polyarea(yz[inds,0], yz[inds,1])
                inds = np.argsort(np.arctan2(z-z.mean(), y-y.mean()))
                area = polyarea(y[inds], z[inds])
                #area = 0.
        self.projected_area = area
        
        return


class Map:
    """Very handy spherical mapping object.
    :INPUTS:
  
      nlon, nlat : scalars
        If mod=='latlon', these inputs specify the number of grid cells
        across map, in latitude and longitude.
  
      inc : scalar
        the inclination, is in units of radians. Zero means we see the
        object equator-on; pi/2 means we see it pole-on.
  
      type : str
        'latlon' or 'eqarea'.
  
      deltaphi : scalar
        Rotation of map, specified in radians.
  
    :OUTPUT:
      A map-class object with various useful fields. Most of these
      fields refer to the coordinates (either Cartesian or spherical
      polar) or the projected radial velocities at the corners of
      specified grid cells, or the approximate projected areas of
      these grid cells.

    :NOTES:
      I have *not* been as careful as I should be in this code -- my
      original goal was speed rather than exactitude.  This means that
      some values are returned as 'nan', and the projected areas are
      only roughly correct.  There's plenty of room for improvement!
        """
    # 2013-05-29 09:37 IJMC: Created
    # 2013-08-07 11:05 IJMC: Added mu field for maps and cells
    # 2014-08-07 15:00 IJMC: Updated documentation -- exactly 1 year later!

    def __init__(self, nlon=20, nlat=10, type='latlon', deltaphi=0, inc=0, verbose=False):
        self.type = type
        self.nlon = nlon
        self.nlat = nlat
        self.ncell = nlon*nlat
        if self.type == "eqarea":
            phi, theta, height, widths, ncell_true = make_eqarea_grid(self.ncell, verbose=verbose)
            self.ncell = ncell_true
        self.deltaphi = deltaphi
        self.inc = inc
        self.cells = []
        self.visible_corners = np.zeros((self.ncell, 3, 4), dtype=float)
        self.corners = np.zeros((self.ncell, 3, 4), dtype=float) # corners in xyz
        self.corners_latlon = np.zeros((self.ncell, 2, 4), dtype=float) # must be the latlon before rot to get correct area
        self.rvcorners = np.zeros((self.ncell, 4), dtype=float) # corners in y direction
        # (0-1) proportional to the projected radial velocity at y coord of that corner
        # rvcorners / np.cos(inc) * vsini [km/s] = rv 
        self.visible_rvcorners = np.zeros((self.ncell, 4), dtype=float) # replace the non-visible by nan
        self.projected_area = np.zeros(self.ncell, dtype=float)
        self.mu = np.zeros(self.ncell, dtype=float)
        self.phi = np.zeros(self.ncell)
        self.theta = np.zeros(self.ncell)

        rot_matrix = np.array([
            [np.cos(inc),    0, -np.sin(inc)], 
            [          0,    1,            0], 
            [np.sin(inc),    0,  np.cos(inc)]
        ])

        if self.type == 'latlon':
            ### Initialize coordinate system:
            #phi0 = np.arange(0, self.nlon+1) * (2*np.pi/self.nlon)
            #theta0 = np.arange(0, self.nlat+1) * (np.pi/self.nlat)
            #phi, theta = np.meshgrid(phi0, theta0)
            #print(phi.shape, theta.shape)
            phi, theta = make_latlon_grid(nlon, nlat)

            ### Rotate by deltaPhi:
            phi1 = (phi + deltaphi).ravel()
            theta1 = theta.ravel()

            ### Convert to x1, y1, z1:
            xyz1 = np.vstack((np.sin(theta1) * np.cos(phi1), \
                                np.sin(theta1) * np.sin(phi1), \
                                np.cos(theta1)))
            ### Rotate by inclination angle i:
            xyz2 = np.dot(rot_matrix, xyz1)
            xyz3 = xyz2.reshape(3, nlat+1, nlon+1)

            kk = 0
            for ii in range(self.nlat):
                for jj in range(self.nlon):
                    cell = MapCell()
                    cell.corners = xyz3[:, ii:ii+2, jj:jj+2].reshape(3,4)
                    cell.corners_latlon = np.vstack((theta[ii:ii+2,jj:jj+2].ravel(), phi[ii:ii+2,jj:jj+2].ravel()))
                    cell.rvcorners = xyz3[1,ii:ii+2,jj:jj+2].ravel() * np.cos(inc)
                    cell.get_projected_area(inc)
                    cell.get_mu()
                    cell.visible_rvcorners = cell.visible_corners[1] * np.cos(inc)

                    self.cells.append(cell)
                    self.corners[kk] = cell.corners
                    self.visible_corners[kk] = cell.visible_corners
                    self.projected_area[kk] = cell.projected_area
                    self.mu[kk] = cell.mu
                    self.corners_latlon[kk] = cell.corners_latlon
                    self.rvcorners[kk] = cell.rvcorners
                    self.visible_rvcorners[kk] = cell.visible_rvcorners
                    kk += 1

        elif self.type == 'eqarea':
            self.nlat = len(theta)
            self.nlon = np.array([len(row) for row in phi])

            phi_corners = [np.zeros((4, self.nlon[m])) for m in range(self.nlat)]
            theta_corners = np.zeros((4, self.nlat))

            for m in range(self.nlat):
                for n in range(self.nlon[m]):
                    phi_corners[m][:,n] = np.array([
                        phi[m][n]-widths[m]/2, phi[m][n]+widths[m]/2, # corner 0, 1
                        phi[m][n]-widths[m]/2, phi[m][n]+widths[m]/2  # corner 2, 3
                    ])

                    theta_corners[:,m] = np.array([
                        theta[m]-height/2, theta[m]-height/2, # corner 0, 1
                        theta[m]+height/2, theta[m]+height/2  # corner 2, 3
                    ])

            ### Rotate by deltaPhi:
            phi_corners_2d = np.concatenate([phi_corners[m] for m in range(self.nlat)], axis=1)
            phi_corners_2d_rot = np.concatenate([phi_corners[m] + deltaphi for m in range(self.nlat)], axis=1)
            theta_corners_2d = np.concatenate([np.tile(theta_corners[:,m], (self.nlon[m], 1)).T for m in range(self.nlat)], axis=1)
            
            phi_corners_1d = phi_corners_2d_rot.ravel()
            theta_corners_1d = theta_corners_2d.ravel()

            ### Convert to x1, y1, z1:
            xyz_1d = np.stack((
                np.sin(theta_corners_1d) * np.cos(phi_corners_1d),
                np.sin(theta_corners_1d) * np.sin(phi_corners_1d),
                np.cos(theta_corners_1d)
            ))

            ### Rotate by inclination angle i:
            xyz_1d_rot = np.dot(rot_matrix, xyz_1d)
            xyz_2d = xyz_1d_rot.reshape(3, 4, ncell_true)

            start = 0
            xyz_3d = []
            theta_corners_3d = []
            phi_corners_3d = []
            for m in range(self.nlat):
                xyz_3d.append(xyz_2d[:, :, start:start+self.nlon[m]])
                theta_corners_3d.append(theta_corners_2d[:, start:start+self.nlon[m]])
                phi_corners_3d.append(phi_corners_2d[:, start:start+self.nlon[m]])
                start = start + self.nlon[m]
            
            kk=0
            for m in range(self.nlat):
                for n in range(self.nlon[m]):
                    cell = MapCell()
                    cell.corners = xyz_3d[m][:,:,n]
                    cell.corners_latlon = np.vstack([theta_corners_3d[m][:,n], phi_corners_3d[m][:,n]])
                    cell.rvcorners = xyz_3d[m][1,:,n] * np.cos(inc)
                    cell.get_projected_area(inc)
                    cell.get_mu()
                    cell.visible_rvcorners = cell.visible_corners[1] * np.cos(inc)

                    self.cells.append(cell)
                    self.corners[kk] = cell.corners
                    self.visible_corners[kk] = cell.visible_corners
                    self.projected_area[kk] = cell.projected_area
                    self.mu[kk] = cell.mu
                    self.corners_latlon[kk] = cell.corners_latlon
                    self.rvcorners[kk] = cell.rvcorners
                    self.visible_rvcorners[kk] = cell.visible_rvcorners
                    kk += 1

        return None

    def get_vprofile(self, v):
        """Compute velocity profile for normalized velocity values v.

        :INPUTS:
           v : NumPy array
             Velocity normalized by the maximum rotation velocity
             observed; i.e., to convert v to true velocities, multiply
             by 2piR/P.
             """ 
        # 2013-05-29 12:28 IJMC: Created
        profile = np.zeros(v.shape, dtype=float)
        for ii in range(self.ncell): # EB: xrange to range
            vmin, vmax = self.visible_rvcorners[ii].min(), self.visible_rvcorners[ii].max()
            profile[(v > vmin) * (v <= vmax)] += 1 #self.projected_area[ii]

        return profile

    def plot_map_cells(self):
        '''Plot the map cells on a Mollweide projection.'''
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.grid(True)
        good = (self.projected_area>0)
        for k in range(self.ncell):
            lats = self.corners_latlon[k][0]
            lons = self.corners_latlon[k][1]

            y = np.array([lats[0], lats[1], lats[3], lats[2]]) - np.pi/2
            x = np.array([lons[0], lons[1], lons[3], lons[2]]) - np.pi
            # Plot the polygon
            if good[k]:
                poly = plt.Polygon(np.column_stack((x, y)), facecolor='gray', edgecolor='black')
                ax.add_patch(poly)
                ax.text(x.mean(), y.mean(), f"{k}", size=5)

        # Set plot parameters
        ax.set_xticklabels([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], fontsize=8)
