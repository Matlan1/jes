from hashlib import sha256
import math

class SpeciesInfo:
    __slots__ = ('sim', 'speciesID', 'ancestorID', 'level', 'apex_pop', 'reps',
                 'prominent', 'coor', 'birth_gen', 'best_fitness',
                 'last_improvement_gen', 'last_seen_gen')

    def __init__(self, _sim, me, ancestor):
        self.sim = _sim
        self.speciesID = me.species
        self.ancestorID = None
        self.level = 0
        if ancestor is not None:
            self.ancestorID = ancestor.species
            self.level = self.sim.species_info[ancestor.species].level+1

        self.apex_pop = 0
        self.reps = [0, 0, 0, 0] # Representative ancestor, first, apex, and last creatures of this species.
        self.prominent = False

        if ancestor is not None:
            self.reps[0] = ancestor.IDNumber
        self.reps[1] = me.IDNumber
        self.coor = None

        self.birth_gen = math.floor(me.IDNumber // self.sim.c_count)
        self.best_fitness = -1e9
        self.last_improvement_gen = self.birth_gen
        self.last_seen_gen = self.birth_gen

    def record_fitness(self, current_gen, fitness):
        if fitness is not None and fitness > self.best_fitness:
            self.best_fitness = fitness
            self.last_improvement_gen = current_gen

    def is_incubating(self, current_gen, leader_tenure=0, pop_median=0.0, pop_max=1.0):
        if self.birth_gen == 0:
            return False
        age = current_gen - self.birth_gen
        if age <= 0:
            return True
        
        # Performance relative to current population
        rel_to_median = self.best_fitness / max(pop_median, 1e-6)
        rel_to_max = self.best_fitness / max(pop_max, 1e-6)
        
        # Tiered Grace Period:
        # Top Newcomers (>= 60% of median or >= 35% of max): Extended VIP grace (25 to 50 gens)
        # Moderate Performers (>= 25% of median): Standard grace (12 to 20 gens)
        # Lowest Performers / Flops: Minimum trial grace (6 to 8 gens)
        if rel_to_median >= 0.6 or rel_to_max >= 0.35:
            max_grace = min(50, 25 + int(0.04 * leader_tenure))
        elif rel_to_median >= 0.25:
            max_grace = min(20, 12 + int(0.02 * leader_tenure))
        else:
            max_grace = 7 # Minimum probationary grace for lowest performers
            
        if age <= max_grace:
            return True
            
        # Active improvement bonus: if the species is actively breaking records, extend grace
        stagnation = current_gen - self.last_improvement_gen
        if stagnation <= 5 and age <= max_grace + 15:
            return True
            
        return False
        
    def becomeProminent(self):  # if you are prominent, all your ancestors become prominent.
        self.prominent = True
        self.sim.pinSpeciesReps(self) # keep our representative creatures' bodies resident forever
        self.insertIntoProminentSpeciesList()
        if self.ancestorID is not None: # you have a parent
            ancestor = self.sim.species_info[self.ancestorID]
            if not ancestor.prominent:
                ancestor.becomeProminent()
                
    def insertIntoProminentSpeciesList(self):
        i = self.speciesID
        p = self.sim.prominent_species
        while len(p) <= self.level: # this level doesn't exist yet. Add new levels of the genealogy tree to acommodate you
            p.append([])
        pL = p[self.level]
        insert_index = 0
        for index in range(len(pL)):  # inefficient sorting thing, but there are <50 species so who cares
            other = pL[index]
            ancestorCompare = 0 if self.level == 0 else self.sim.species_info[other].ancestorID-self.ancestorID
            if ancestorCompare == 0: #siblings
                if other < i:
                    insert_index = index+1
            else: #not siblings trick to avoid family trees tangling (all siblings should be adjacent)
                if ancestorCompare < 0:
                    insert_index = index+1
        pL.insert(insert_index,i)

    def getWhen(self, index):
        return math.floor(self.reps[index]//self.sim.c_count)
        
    def getPerformance(self, sim, index):
        creature = sim.getCreatureWithID(self.reps[index])
        if creature is None:
            return None # representative's generation was retired past a checkpoint
        return creature.fitness
        
        