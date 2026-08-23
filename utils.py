from hashlib import sha256
import numpy as np
import math

def arrayLerp(arrA, arrB, x):
    return arrA+(arrB-arrA)*x
    
def listLerp(listA, listB, x):
    listResult = [None]*len(listA)
    for i in range(len(listA)):
        listResult[i] = listA[i]+(listB[i]-listA[i])*x
    return listResult
    
def getUnit(r):
    _list = [0.000001,0.000002,0.000005,0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000]
    choice = 0
    while(_list[choice] < r*0.2):
        choice += 1
    return _list[choice]
    
def hue_to_rgb(hue):
    h = (hue*6)%1.0
    r = (0,0,0)
    if hue < 1/6:
        r = (1,h,0)
    elif hue < 2/6:
        r = (1-h,1,0)
    elif hue < 3/6:
        r = (0,1,h)
    elif hue < 4/6:
        r = (h*0.4,1-h*0.6,1)
    elif hue < 5/6:
        r = (0.4+0.6*h,0.4,1)
    else:
        r = (1,0.4-0.4*h,1-h)
    return (255*r[0], 200*r[1],255*r[2])
    
def species_to_name(s, ui):
    salted = str(s)+ui.salt
    _hex = sha256(salted.encode('utf-8')).hexdigest()
    result = int(_hex, 16)
    length_choices = [5,5,6,6,7]
    length_choice = result%5
    result = result//5
    
    letters = ["bcdfghjklmnprstvwxz","aeiouy"]
    name_len = length_choices[length_choice]
    name = ""
    for n in range(name_len):
        letter_type = n%len(letters)
        option_count = len(letters[letter_type])
        choice = result%option_count
        letter = letters[letter_type][choice]
        if n >= 2 and letter == "g" and name[n-2].lower() == "n":
            letter = "m"
        if n == 0:
            letter = letter.upper()
        name += letter
        result = result//option_count
    return name
    
def brighten(color, b):
    if b >= 1:
        fac = b-1
        return (lerp(color[0],255,fac),lerp(color[1],255,fac),lerp(color[2],255,fac))
    else:
        return (color[0]*b, color[1]*b, color[2]*b)
    
def speciesToColor(s, ui):
    salted = str(s)+ui.salt
    if s in ui.sc_colors:
        salted = ui.sc_colors[s]+ui.salt
    _hex = sha256(salted.encode('utf-8')).hexdigest()
    hue = (int(_hex, 16)%10000)/10000
    brightness = (math.floor(int(_hex, 16)//10000)%100)/100
    color = hue_to_rgb(hue)
    new_color = brighten(color, 0.85+0.6*brightness)
    return new_color
    
def bound(x):
    return min(max(x,0),1)
    
def lerp(a,b,x):
    return a+(b-a)*x
    
def dist_to_text(dist, sigfigs, u):
    if sigfigs:
        return f"{dist/u:.2f}cm"
    else:
        return str(int(dist/u))+"cm"
        
def getDistanceArray(a,b):
    x_dist = a[:,:,:,0]-b[:,:,:,0]
    y_dist = a[:,:,:,1]-b[:,:,:,1]
    return np.sqrt(np.square(x_dist)+np.square(y_dist))
    
def applyMuscles(n,m,muscle_coef):
    xNeighborDists = getDistanceArray(n[:,:-1,:],n[:,1:,:])
    yNeighborDists = getDistanceArray(n[:,:,:-1],n[:,:,1:])
    posDiagNeighborDists = getDistanceArray(n[:,:-1,:-1],n[:,1:,1:])
    negDiagNeighborDists = getDistanceArray(n[:,:-1,1:],n[:,1:,:-1])
    
    MAs = [None]*6
    segments = [[0,0,1,0],[0,1,1,1],
    [0,0,0,1],[1,0,1,1],[0,0,1,1],[0,1,1,0]]
    segments2 = [[0,0,1,0],[0,1,1,1],
    [0,0,0,1],[1,0,1,1],[0,0,1,1],[0,1,1,0]]
    
    
    MAs[0] = getMuscleAttraction(xNeighborDists[:,:,:-1],m[:,:,:,0],muscle_coef)
    MAs[1] = getMuscleAttraction(xNeighborDists[:,:,1:],m[:,:,:,0],muscle_coef)
    MAs[2] = getMuscleAttraction(yNeighborDists[:,:-1,:],m[:,:,:,1],muscle_coef)
    MAs[3] = getMuscleAttraction(yNeighborDists[:,1:,:],m[:,:,:,1],muscle_coef)
    MAs[4] = getMuscleAttraction(posDiagNeighborDists,m[:,:,:,3],muscle_coef)
    MAs[5] = getMuscleAttraction(negDiagNeighborDists,m[:,:,:,3],muscle_coef)
    
    # The array n is a 100 x 5 x 5 x 4 dimensional array,
    # and it encodes the position and velocity data for all 100 creatures on a frame.
    
    # Dimension 1: 100 creatures (creature ID)
    # Dimension 2: 5 nodes across the x-dimensional
    # Dimension 3: 5 nodes across the y-dimensional
    # Dimension 4: Which coordinate to do you want (x, y, vx, vy)
    _, CW, CH, __ = n.shape
    CW -= 1
    CH -= 1
    
    for dire in range(6):
        s = segments[dire]
        sli1 = n[:,s[0]:s[0]+CW,s[1]:s[1]+CW]
        sli2 = n[:,s[2]:s[2]+CH,s[3]:s[3]+CH]
        
        delta_x = sli1[:,:,:,0]-sli2[:,:,:,0]
        delta_y = sli1[:,:,:,1]-sli2[:,:,:,1]
        
        delta_magnitude = np.sqrt(np.square(delta_x)+np.square(delta_y))
        delta_nx = delta_x/delta_magnitude
        delta_ny = delta_y/delta_magnitude
        
        n[:,s[0]:s[0]+CW,s[1]:s[1]+CW,2] += delta_nx*MAs[dire]
        n[:,s[0]:s[0]+CW,s[1]:s[1]+CW,3] += delta_ny*MAs[dire]
        n[:,s[2]:s[2]+CH,s[3]:s[3]+CH,2] -= delta_nx*MAs[dire]
        n[:,s[2]:s[2]+CH,s[3]:s[3]+CH,3] -= delta_ny*MAs[dire]
        
def getMuscleAttraction(dists,m,muscle_coef):
    return (m-dists)*muscle_coef
    
def getDist(x1, y1, x2, y2):
    return np.linalg.norm(np.array([x2-x1,y2-y1]))
    
def arrayIntMultiply(arr, factor):
    result = [None]*len(arr)
    for i in range(len(arr)):
        result[i] = int(arr[i]*factor)
    return result

def flipDNA(dna, CW, CH, beats_per_cycle, traits_per_box):
    """
    Horizontally mirrors a creature's DNA matrix across its width dimension (CW).
    Transforms a leftward sprinter into an identical rightward sprinter.
    """
    DNA_LEN = CH * CW * beats_per_cycle * traits_per_box
    grid = dna[:DNA_LEN].reshape(CH, CW, beats_per_cycle, traits_per_box)
    flipped_grid = np.flip(grid, axis=1) # flip horizontally along CW
    new_dna = dna.copy()
    new_dna[:DNA_LEN] = flipped_grid.flatten()
    return new_dna

def parse_population_size(raw_input, default=250):
    """
    Parses and autocorrects input population size to the nearest valid value.
    Valid population sizes are even positive integers >= 2.
    """
    if raw_input is None:
        return default
    text = str(raw_input).strip()
    if not text:
        return default
    try:
        val = float(text)
    except ValueError:
        print(f"Invalid input '{text}'. Autocorrecting to default population size: {default}")
        return default

    if not math.isfinite(val):
        print(f"Invalid input '{text}'. Autocorrecting to default population size: {default}")
        return default

    int_val = round(val)
    if int_val < 2:
        corrected = 2
        print(f"Population size must be at least 2. Autocorrecting {raw_input} -> {corrected}")
        return corrected

    if int_val % 2 != 0:
        corrected = int_val + 1
        print(f"Population size must be an even integer for pair-wise selection. Autocorrecting {raw_input} -> {corrected}")
        return corrected

    if int_val != val:
        print(f"Autocorrecting float {raw_input} -> {int_val}")
    return int_val

def get_mosaic_dim(c_count):
    """
    Computes optimal [big_icons, small_icons, species_tiles, lightboard] grid columns
    for any arbitrary population size.
    """
    if c_count == 250:
        return [10, 24, 24, 30]
    if c_count <= 20:
        val = max(2, c_count)
        return [val, val, val, max(2, val)]
    cols_small = max(4, math.ceil(math.sqrt(c_count * 2.3)))
    cols_big = max(2, int(cols_small * 0.42))
    cols_lb = max(4, math.ceil(math.sqrt(c_count * 3.6)))
    return [cols_big, cols_small, cols_small, cols_lb]

# Multi-Core Parallel JIT Engine
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def jit_simulateRun(nodeCoor_in, muscles, startCurrentFrame, frameCount, calmingRun,
                        beat_time, beats_per_cycle, gravity, calming_fric, typical_fric,
                        ground_fric, muscle_coef, ceiling_y, floor_y):
        COUNT, H_nodes, W_nodes, _ = nodeCoor_in.shape
        CH = H_nodes - 1
        CW = W_nodes - 1
        nodeCoor = nodeCoor_in.copy()
        friction = calming_fric if calmingRun else typical_fric

        for c in prange(COUNT):
            for f in range(frameCount):
                currentFrame = startCurrentFrame + f
                beat = 0
                if not calmingRun:
                    beat = (currentFrame // beat_time) % beats_per_cycle
                    for ny in range(H_nodes):
                        for nx in range(W_nodes):
                            nodeCoor[c, ny, nx, 3] += gravity

                for cy in range(CH):
                    for cx in range(CW):
                        m0 = muscles[c, cy, cx, beat, 0]
                        m1 = muscles[c, cy, cx, beat, 1]
                        m3 = muscles[c, cy, cx, beat, 3]

                        # Seg 0: (cy, cx) to (cy+1, cx)
                        dx = nodeCoor[c, cy, cx, 0] - nodeCoor[c, cy+1, cx, 0]
                        dy = nodeCoor[c, cy, cx, 1] - nodeCoor[c, cy+1, cx, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m0 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy, cx, 2] += fx
                            nodeCoor[c, cy, cx, 3] += fy
                            nodeCoor[c, cy+1, cx, 2] -= fx
                            nodeCoor[c, cy+1, cx, 3] -= fy

                        # Seg 1: (cy, cx+1) to (cy+1, cx+1)
                        dx = nodeCoor[c, cy, cx+1, 0] - nodeCoor[c, cy+1, cx+1, 0]
                        dy = nodeCoor[c, cy, cx+1, 1] - nodeCoor[c, cy+1, cx+1, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m0 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy, cx+1, 2] += fx
                            nodeCoor[c, cy, cx+1, 3] += fy
                            nodeCoor[c, cy+1, cx+1, 2] -= fx
                            nodeCoor[c, cy+1, cx+1, 3] -= fy

                        # Seg 2: (cy, cx) to (cy, cx+1)
                        dx = nodeCoor[c, cy, cx, 0] - nodeCoor[c, cy, cx+1, 0]
                        dy = nodeCoor[c, cy, cx, 1] - nodeCoor[c, cy, cx+1, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m1 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy, cx, 2] += fx
                            nodeCoor[c, cy, cx, 3] += fy
                            nodeCoor[c, cy, cx+1, 2] -= fx
                            nodeCoor[c, cy, cx+1, 3] -= fy

                        # Seg 3: (cy+1, cx) to (cy+1, cx+1)
                        dx = nodeCoor[c, cy+1, cx, 0] - nodeCoor[c, cy+1, cx+1, 0]
                        dy = nodeCoor[c, cy+1, cx, 1] - nodeCoor[c, cy+1, cx+1, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m1 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy+1, cx, 2] += fx
                            nodeCoor[c, cy+1, cx, 3] += fy
                            nodeCoor[c, cy+1, cx+1, 2] -= fx
                            nodeCoor[c, cy+1, cx+1, 3] -= fy

                        # Seg 4: (cy, cx) to (cy+1, cx+1)
                        dx = nodeCoor[c, cy, cx, 0] - nodeCoor[c, cy+1, cx+1, 0]
                        dy = nodeCoor[c, cy, cx, 1] - nodeCoor[c, cy+1, cx+1, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m3 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy, cx, 2] += fx
                            nodeCoor[c, cy, cx, 3] += fy
                            nodeCoor[c, cy+1, cx+1, 2] -= fx
                            nodeCoor[c, cy+1, cx+1, 3] -= fy

                        # Seg 5: (cy, cx+1) to (cy+1, cx)
                        dx = nodeCoor[c, cy, cx+1, 0] - nodeCoor[c, cy+1, cx, 0]
                        dy = nodeCoor[c, cy, cx+1, 1] - nodeCoor[c, cy+1, cx, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1e-9:
                            ma = (m3 - dist) * muscle_coef / dist
                            fx = dx * ma
                            fy = dy * ma
                            nodeCoor[c, cy, cx+1, 2] += fx
                            nodeCoor[c, cy, cx+1, 3] += fy
                            nodeCoor[c, cy+1, cx, 2] -= fx
                            nodeCoor[c, cy+1, cx, 3] -= fy

                for ny in range(H_nodes):
                    for nx in range(W_nodes):
                        nodeCoor[c, ny, nx, 2] *= friction
                        nodeCoor[c, ny, nx, 3] *= friction
                        nodeCoor[c, ny, nx, 0] += nodeCoor[c, ny, nx, 2]
                        nodeCoor[c, ny, nx, 1] += nodeCoor[c, ny, nx, 3]

                        if not calmingRun:
                            py = nodeCoor[c, ny, nx, 1]
                            if py >= floor_y:
                                pressure = py - floor_y
                                g_mult = 0.5 ** (pressure * ground_fric)
                                nodeCoor[c, ny, nx, 1] = floor_y
                                nodeCoor[c, ny, nx, 2] *= g_mult

            if calmingRun:
                mean_x = 0.0
                for ny in range(H_nodes):
                    for nx in range(W_nodes):
                        mean_x += nodeCoor[c, ny, nx, 0]
                mean_x /= (H_nodes * W_nodes)
                for ny in range(H_nodes):
                    for nx in range(W_nodes):
                        nodeCoor[c, ny, nx, 0] -= mean_x

        return nodeCoor
else:
    jit_simulateRun = None