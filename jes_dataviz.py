import numpy as np
from utils import getUnit, dist_to_text, species_to_name, speciesToColor
from jes_shapes import centerText, rightText, alignText, drawArrow, drawSpeciesCircle
import math
import pygame
import random
import bisect

def get_plot_indices(total_gens, graph_width):
    """Return a sorted list of generation indices to plot.
    
    Recent 25% of generations: always full resolution (every gen).
    Older 75%: gradually thinned — denser near the recent boundary,
    sparser at the very beginning.  When total_gens fits comfortably
    in graph_width pixels, every generation is kept as-is.
    """
    if total_gens <= 1:
        return list(range(total_gens))

    # If everything fits at ≥1 px per gen, no thinning needed
    if total_gens <= graph_width:
        return list(range(total_gens))

    recent_frac = 0.25
    recent_start = int(total_gens * (1.0 - recent_frac))   # first index of "recent" zone
    recent_indices = list(range(recent_start, total_gens))  # always kept

    # How many pixels remain for the old zone?
    budget = max(graph_width - len(recent_indices), 20)

    # Build old indices with inverse-quadratic spacing:
    #   t in [0,1]  →  idx = (1-(1-t)²) * recent_start
    # Sparse at gen 0, dense approaching the recent boundary.
    old_set = {0}   # always include the very first generation
    for i in range(budget):
        t = i / max(budget - 1, 1)
        idx = int((1.0 - (1.0 - t) ** 2) * (recent_start - 1))
        old_set.add(idx)

    return sorted(old_set | set(recent_indices))

def drawAllGraphs(sim, ui):
    drawLineGraph(sim.percentiles, ui.graph, [70,0,30,30], sim.UNITS_PER_METER, ui.smallFont)
    drawSAC(sim.species_pops, ui.sac, [70,0], ui)
    drawGeneGraph(sim.species_info, sim.prominent_species, ui.gene_graph, sim, ui, ui.tinyFont)

def drawLineGraph(data,graph,margins,u,font):
    BLACK = (0,0,0)
    GRAY25 = (70,70,70)
    GRAY50 = (128,128,128)
    WHITE = (255,255,255)
    RED = (255,0,0)

    graph.fill(BLACK)
    W = graph.get_width()-margins[0]-margins[1]
    H = graph.get_height()-margins[2]-margins[3]
    LEFT = margins[0]
    RIGHT = graph.get_width()-margins[1]
    BOTTOM = graph.get_height()-margins[3]
    
    LEN = len(data)
    if LEN == 0:
        return
        
    minVal = min(0.0, min(float(d[-1]) for d in data))
    maxVal = max(1e-6, max(float(d[0]) for d in data))
    denom = max(maxVal - minVal, 1e-6)
    unit = getUnit((maxVal-minVal)/u)*u
    tick = math.floor(minVal/unit)*unit-unit
    while tick <= maxVal+unit:
        ay = float(BOTTOM-H*(tick-minVal)/denom)
        pygame.draw.line(graph, GRAY25, (LEFT, ay), (RIGHT, ay), width=1)
        rightText(graph, dist_to_text(tick, False, u), LEFT-7, ay, GRAY50, font)
        tick += unit
        
    toShow = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,91,92,93,94,95,96,97,98,99,100]
    # Downsample old history the same way the SAC does; recent 25% stays full-res.
    indices = get_plot_indices(LEN, W)
    for p in toShow:
        prev_x = None
        prev_y = None
        for g in indices:
            nextVal = float(data[g][p])

            x2 = float(LEFT+((g+1)/LEN)*W)
            y2 = float(BOTTOM-H*(nextVal-minVal)/denom)

            if prev_x is not None:
                IMPORTANT = (p%10 == 0)
                thickness = 2 if IMPORTANT else 1
                color = WHITE if IMPORTANT else GRAY50
                if p == 50:
                    color = RED
                    thickness = 3
                pygame.draw.line(graph, color, (prev_x, prev_y), (x2, y2), width=thickness)

            prev_x = x2
            prev_y = y2
            
def getRangeEvenIfNone(dicty, sorted_keys, key):
    if key in dicty:
        return dicty[key]
    n = bisect.bisect(sorted_keys, key+0.5)
    if n >= len(sorted_keys):
        val = dicty[sorted_keys[n-1]][2]
    else:
        val = dicty[sorted_keys[n]][1]
    return [0,val,val]

def trapezoidHelper(sac, data, iter_keys, sorted_keys, g1, g2, i_start, i_end, x1, x2, FAC, level, ui, color_map):
    pop2 = [0,0,0]
    H = sac.get_height()
    d1 = data[g1]
    d2 = data[g2]
    sk2 = sorted_keys[g2]
    
    for sp in iter_keys[g1]:
        pop1 = d1[sp]
        if level == 1:
            if pop1[2] <= i_start or pop1[1] >= i_end:
                continue
        if level == 0 and pop1[1] != pop2[2]:
            gap_start = min(pop2[2], pop1[1])
            gap_end = max(pop2[2], pop1[1])
            trapezoidHelper(sac, data, iter_keys, sorted_keys, g2, g1, gap_start, gap_end, x2, x1, FAC, 1, ui, color_map)
        pop2 = getRangeEvenIfNone(d2, sk2, sp)
        points = [[x1,H-pop2[1]*FAC],[x1,H-pop2[2]*FAC],[x2,H-pop1[2]*FAC],[x2,H-pop1[1]*FAC]]
        pygame.draw.polygon(sac, color_map[sp], points)

class IncrementalSACRenderer:
    def __init__(self):
        self.cached_surface = None
        self.cached_gen = -1
        self.color_map = {}

    def _full_redraw(self, data, sac, W, H, LEFT, LEN, ui):
        """Fast full redraw: fill pixel columns directly via numpy array."""
        sac.fill((0,0,0))

        indices = get_plot_indices(LEN, W)

        # Get direct pixel access (H x W x 3 via surfarray is [x, y, channel])
        pixels = pygame.surfarray.pixels3d(sac)

        for k in range(len(indices)):
            g = indices[k]

            # Pixel column range for this plotted generation
            if k > 0:
                x1 = int(LEFT + (indices[k-1] + 1) / LEN * W)
            else:
                x1 = int(LEFT)
            x2 = int(LEFT + (g + 1) / LEN * W)
            if x2 <= x1 or x1 >= pixels.shape[0] or x2 <= 0:
                continue
            x1 = max(x1, 0)
            x2 = min(x2, pixels.shape[0])

            sp_data = data[g]
            sorted_sps = sorted(sp_data.keys())
            if not sorted_sps:
                continue
            c_count = sp_data[sorted_sps[-1]][2]
            if c_count == 0:
                continue

            for sp in sorted_sps:
                pop = sp_data[sp]
                y_top = max(0, int(H - pop[2] / c_count * H))
                y_bot = min(H, int(H - pop[1] / c_count * H))
                if y_bot <= y_top:
                    continue
                if sp not in self.color_map:
                    self.color_map[sp] = speciesToColor(sp, ui)
                color = self.color_map[sp]
                pixels[x1:x2, y_top:y_bot] = (int(color[0]), int(color[1]), int(color[2]))

        del pixels  # release surfarray lock
        self.cached_surface = sac.copy()
        self.cached_gen = LEN - 1

    def draw(self, data, sac, margins, ui):
        LEN = len(data)
        if LEN == 0:
            return
        W = sac.get_width() - margins[0] - margins[1]
        H = sac.get_height()
        LEFT = margins[0]

        # Already up to date — just show cache
        if self.cached_surface is not None and self.cached_gen == LEN - 1:
            sac.blit(self.cached_surface, (0, 0))
            return

        # First draw or non-sequential jump (e.g. slider scrub) → full redraw
        if self.cached_surface is None or self.cached_gen != LEN - 2:
            self._full_redraw(data, sac, W, H, LEFT, LEN, ui)
            return

        # Sequential step (cached_gen == LEN - 2).
        # When each gen still gets ≥1 pixel, use the fast incremental path.
        if LEN <= W:
            self._incremental(data, sac, W, H, LEFT, LEN, ui)
            return

        # Past pixel width — periodic compaction cycle.
        # Redraw every N gens so the chart stays readable.
        redraw_interval = max(5, LEN // 100)
        gens_since = LEN - 1 - self.cached_gen
        if gens_since < redraw_interval:
            # Between redraws, just show the cached surface
            sac.blit(self.cached_surface, (0, 0))
            return

        self._full_redraw(data, sac, W, H, LEFT, LEN, ui)

    def _incremental(self, data, sac, W, H, LEFT, LEN, ui):
        """Fast incremental: scale previous surface and draw one new gen slice."""
        g = LEN - 1
        prev_w = int(W * (g / LEN))
        prev_area = pygame.Rect(LEFT, 0, W, H)
        scaled_prev = pygame.transform.scale(self.cached_surface.subsurface(prev_area), (prev_w, H))
        
        sac.fill((0,0,0))
        sac.blit(scaled_prev, (LEFT, 0))

        iter_keys = [list(data[g-1].keys()), list(data[g].keys())]
        sorted_keys = [sorted(iter_keys[0]), sorted(iter_keys[1])]
        
        for sp in iter_keys[0] + iter_keys[1]:
            if sp not in self.color_map:
                self.color_map[sp] = speciesToColor(sp, ui)

        x1 = LEFT + prev_w
        x2 = LEFT + W
        if not sorted_keys[1]:
            self.cached_surface = sac.copy()
            self.cached_gen = LEN - 1
            return
        c_count = data[g][sorted_keys[1][-1]][2]
        if c_count == 0:
            self.cached_surface = sac.copy()
            self.cached_gen = LEN - 1
            return
        FAC = H / c_count
        
        d_slice = [data[g-1], data[g]]
        trapezoidHelper(sac, d_slice, iter_keys, sorted_keys, 1, 0, 0, c_count, x1, x2, FAC, 0, ui, self.color_map)
        self.cached_surface = sac.copy()
        self.cached_gen = LEN - 1

_sac_renderer = IncrementalSACRenderer()

def drawSAC(data, sac, margins, ui):
    _sac_renderer.draw(data, sac, margins, ui)
        
def drawGeneGraph(species_info, ps, gg, sim, ui, font):  # ps = prominent_species
    R = ui.GENEALOGY_COOR[4]
    H = gg.get_height()-R*2
    W = gg.get_width()-R*2
    gg.fill((0,0,0))
    if len(sim.creatures) == 0:
        return
        
    for level in range(len(ps)):
        for i in range(len(ps[level])):
            s = ps[level][i]
            x = (i+0.5)/(len(ps[level]))*W+R
            y = (level)/(len(ps)-0.8)*H+R
            species_info[s].coor = (x,y)
            
    for level in range(len(ps)):
        for i in range(len(ps[level])):
            s = ps[level][i]
            drawSpeciesCircle(gg,s,species_info[s].coor,R,sim,species_info,font,True,ui)
        
def displayAllGraphs(screen, sim, ui):
    WHITE = (255,255,255)
    blitGraphsandMarks(screen, sim, ui)
    blitGGandMarks(screen, sim, ui)
    
    if sim.last_gen_run_time >= 0:
        rightText(screen, f"Last gen runtime: {sim.last_gen_run_time:.3f}s", 1200,28, WHITE, ui.smallFont)
        
def blitGraphsandMarks(screen, sim, ui):
    screen.blit(ui.graph,ui.GRAPH_COOR[0:2])
    screen.blit(ui.sac,ui.SAC_COOR[0:2])
    GREEN = (0,255,0)
    WHITE = (255,255,255)
    a = int(ui.genSlider.val)
    b = int(ui.genSlider.val_max)
    a2 = min(a,b-1)
    if b == 0:
        return
    
    if a < b:
        frac = (a+1)/b
        lineX = ui.SAC_COOR[0]+70+(ui.graph.get_width()-70)*frac
        lineYs = [[50,550],[560,860]]
        for lineY in lineYs:
            pygame.draw.line(screen, GREEN, (lineX, lineY[0]), (lineX, lineY[1]), width=2)  
    
    frac = (a2+1)/b
    lineX = ui.SAC_COOR[0]+70+(ui.graph.get_width()-70)*frac
    median = sim.percentiles[a2][50]
    rightText(screen, f"Median: {dist_to_text(median, True, sim.UNITS_PER_METER)}", 1800,28, WHITE, ui.smallFont)
    
    top_species = getTopSpecies(sim, a2)
    for sp in sim.species_pops[a2].keys():
        pop = sim.species_pops[a2][sp]
        if pop[0] >= sim.c_count*sim.S_VISIBLE:
            speciesI = (pop[1]+pop[2])/2
            speciesY = 560+300*(1-speciesI/sim.c_count)
            name = species_to_name(sp, ui)
            color = speciesToColor(sp, ui)
            OUTLINE = ui.WHITE if sp == top_species else None
            alignText(screen, f"{name}: {pop[0]}", lineX+10, speciesY, color, ui.smallFont, 0.0, [ui.BLACK,OUTLINE])
        

def blitGGandMarks(screen, sim, ui):
    screen.blit(ui.gene_graph,ui.GENEALOGY_COOR[0:2])
    R = 42
    a = int(ui.genSlider.val)
    b = int(ui.genSlider.val_max)
    a2 = min(a,b-1)
    if b == 0:
        return
    top_species = getTopSpecies(sim, a2)
    
    
    for sp in sim.species_pops[a2].keys():
        info = sim.species_info[sp]
        if not info.prominent:
            continue
        pop = sim.species_pops[a2][sp][0]
        circle_count = 2 if sp == top_species else 1
        cx = info.coor[0]+ui.GENEALOGY_COOR[0]
        cy = info.coor[1]+ui.GENEALOGY_COOR[1]
        for c in range(circle_count):
            pygame.draw.circle(screen, ui.WHITE, (cx,cy), R+3+6*c, 3)
    
    if ui.species_storage is not None:
        sp = ui.species_storage
        if sp in sim.species_pops[a2]:
            circle_count = 2 if sp == top_species else 1
            for c in range(circle_count):
                pygame.draw.circle(screen, ui.WHITE, ui.storage_coor, R+3+6*c, 3)
            
        
    

    
        
def getTopSpecies(sim, g):
    data = sim.species_pops[g] 
    return max(data, key=data.get)