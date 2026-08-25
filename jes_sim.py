import numpy as np
from utils import getDistanceArray, applyMuscles, HAS_NUMBA, jit_simulateRun, flipDNA, N_IN, N_HID, N_OUT, W1_LEN, B1_LEN, W2_LEN, B2_LEN, BRAIN_LEN
from jes_creature import Creature
from jes_species_info import SpeciesInfo
from jes_dataviz import drawAllGraphs
from jes_history import HistoryStore, PopArchiveList, TieredRankings, GenBodyCache
import time
import random

class Sim:
    def __init__(self, _c_count, _stabilization_time, _trial_time, _beat_time,
    _beat_fade_time, _c_dim, _beats_per_cycle, _node_coor_count,
    _y_clips, _ground_friction_coef, _gravity_acceleration_coef,
    _calming_friction_coef, _typical_friction_coef, _muscle_coef,
    _traits_per_box, _traits_extra, _mutation_rate, _big_mutation_rate, _UNITS_PER_METER,
    _ram_window_gens=20, _snapshot_every=100, _lru_capacity=6, _archive_dir="history_archive",
    _pops_tail_gens=2048):
        self.c_count = _c_count #creature count
        self.species_count = _c_count #species count
        self.stabilization_time = _stabilization_time
        self.trial_time = _trial_time
        self.beat_time = _beat_time
        self.beat_fade_time = _beat_fade_time
        self.c_dim = _c_dim
        self.CW, self.CH = self.c_dim
        self.beats_per_cycle = _beats_per_cycle
        self.node_coor_count = _node_coor_count 
        self.y_clips = _y_clips
        self.ground_friction_coef = _ground_friction_coef
        self.gravity_acceleration_coef = _gravity_acceleration_coef
        self.calming_friction_coef = _calming_friction_coef
        self.typical_friction_coef = _typical_friction_coef
        self.muscle_coef = _muscle_coef
        
        self.traits_per_box = _traits_per_box
        self.traits_extra = _traits_extra
        
        # Neural Brain architecture
        self.N_IN = N_IN
        self.N_HID = N_HID
        self.N_OUT = N_OUT
        self.W1_LEN = W1_LEN
        self.B1_LEN = B1_LEN
        self.W2_LEN = W2_LEN
        self.B2_LEN = B2_LEN
        self.BRAIN_LEN = BRAIN_LEN
        self.DNA_MORPH_LEN = self.CW*self.CH*self.beats_per_cycle*self.traits_per_box+self.traits_extra
        self.trait_count = self.DNA_MORPH_LEN + self.BRAIN_LEN
        
        self.mutation_rate = _mutation_rate
        self.big_mutation_rate = _big_mutation_rate
        
        self.S_VISIBLE = 0.05 #what proportion of the population does a species need to appear on the SAC graph?
        self.S_NOTABLE = 0.10 #what proportion of the population does a species need to appear in the genealogy?
        self.HUNDRED = 100 # change this if you want to change the resolution of the percentile-tracking
        self.UNITS_PER_METER = _UNITS_PER_METER
        self.max_species_fraction = 0.50 # Option 2: Anti-monopoly carrying capacity cap (max 50% population per species)
        self.current_leader = None
        self.leader_tenure = 0
        self.min_incubation_survivors = 2 # Option 1: Minimum protected creatures for incubating young species

        # Tiered history: keeps RAM flat on very long runs.
        # Hot window (recent gens) fully resident; checkpoint generations spill their
        # bodies (dna + calmState + rankings) to disk; other old generations keep only
        # chart data (percentiles in RAM, species pops via a disk-backed list).
        self.ram_window_gens = max(2, int(_ram_window_gens))
        self.snapshot_every = max(1, int(_snapshot_every))
        calm_len = (self.CH + 1) * (self.CW + 1) * self.node_coor_count
        self.store = HistoryStore(_archive_dir, self.c_count, self.trait_count, calm_len, self.snapshot_every)
        self.DISK_WARN_FREE_BYTES = 20 * (1024 ** 3)
        self.DISK_MIN_FREE_BYTES = 5 * (1024 ** 3)
        self._last_disk_warning_time = -1e9
        self.archived_upto = 0          # generations [0, archived_upto) have been retired
        self.pinned_creatures = {}      # IDNumber -> Creature whose body stays resident forever
        self.temp_pins = set()          # Creatures protected only while highlighted/movies play
        self._alive_last_gen = set()    # species present in the previous generation's census
        self.body_cache = GenBodyCache(self.store, _lru_capacity,
                                       on_evict=self._onBodyCacheEvict,
                                       is_protected=self._isProtected)

        self.creatures = None
        self.rankings = TieredRankings(self.store, self.ram_window_gens)
        self.percentiles = []
        self.species_pops = PopArchiveList(self.store, _pops_tail_gens)
        self.species_info = []
        self.prominent_species = []
        self.ui = None
        self.last_gen_run_time = -1

    def _isProtected(self, c):
        return c.pinned or c in self.temp_pins

    def _onBodyCacheEvict(self, gen):
        row = self.creatures[gen] if self.creatures is not None and gen < len(self.creatures) else None
        if row is not None:
            self.body_cache.strip_shells(row)
        
    def initializeUniverse(self):
        self.creatures = [[None]*self.c_count]
        for c in range(self.c_count):
            self.creatures[0][c] = self.createNewCreature(c)
            self.species_info.append(SpeciesInfo(self,self.creatures[0][c], None))
            
        # We want to make sure that all creatures, even in their
        # initial state, are in calm equilibrium. They shouldn't
        # be holding onto potential energy (e.g. compressed springs)
        self.getCalmStates(0,0,self.c_count,self.stabilization_time,True) #Calm the creatures down so no potential energy is stored
        self.ui.drawCreatureMosaic(0)
        
    def createNewCreature(self, idNumber):
        dna = np.clip(np.random.normal(0.0, 1.0, self.trait_count).astype(np.float32), -3.0, 3.0)
        return Creature(dna, idNumber, -1, self, self.ui)
        
    def getCalmStates(self, gen, startIndex, endIndex, frameCount, calmingRun):
        param = self.simulateImport(gen, startIndex, endIndex, False)
        simResult = self.simulateRun(param, frameCount, True)
        nodeCoor = simResult[0]
        for c in range(self.c_count):
            self.creatures[gen][c].saveCalmState(nodeCoor[c])
            
    def getStartingNodeCoor(self, gen, startIndex, endIndex, fromCalmState):
        COUNT = endIndex-startIndex
        n = np.zeros((COUNT,self.CH+1,self.CW+1,self.node_coor_count))
        if not fromCalmState or self.creatures[gen][0].calmState is None:
            # create grid of nodes along perfect gridlines
            coorGrid = np.mgrid[0:self.CW+1,0:self.CH+1]
            coorGrid = np.swapaxes(np.swapaxes(coorGrid,0,1),1,2)
            n[:,:,:,0:2] = coorGrid
        else:
            # load calm state into nodeCoor
            for c in range(startIndex,endIndex):
                n[c-startIndex,:,:,:] = self.creatures[gen][c].calmState
            n[:,:,:,1] -= self.CH  # lift the creature above ground level
        return n

    def getMuscleArray(self, gen, startIndex, endIndex):
        return self.buildMuscleArray([self.creatures[gen][c].dna for c in range(startIndex, endIndex)])

    def buildMuscleArray(self, dnas):
        COUNT = len(dnas)
        m = np.empty((COUNT,self.CH,self.CW,self.beats_per_cycle,self.traits_per_box+1)) # add one trait for diagonal length.
        DNA_LEN = self.CH*self.CW*self.beats_per_cycle*self.traits_per_box
        all_dna = np.array([d[:DNA_LEN] for d in dnas]).reshape(COUNT, self.CH, self.CW, self.beats_per_cycle, self.traits_per_box)
        m[:,:,:,:,:self.traits_per_box] = 1.0 + all_dna / 3.0
        m[:,:,:,:,3] = np.hypot(m[:,:,:,:,0], m[:,:,:,:,1]) # Set diagonal tendons
        return m

    def getBrainArrays(self, gen, startIndex, endIndex):
        return self.buildBrainArrays([self.creatures[gen][c].dna for c in range(startIndex, endIndex)])

    def buildBrainArrays(self, dnas):
        COUNT = len(dnas)
        idx = self.DNA_MORPH_LEN
        all_brains = np.array([d[idx:idx+self.BRAIN_LEN] for d in dnas])

        w1_end = self.W1_LEN
        b1_end = w1_end + self.B1_LEN
        w2_end = b1_end + self.W2_LEN
        b2_end = w2_end + self.B2_LEN

        W1 = all_brains[:, :w1_end].reshape(COUNT, self.N_IN, self.N_HID)
        b1 = all_brains[:, w1_end:b1_end].reshape(COUNT, self.N_HID)
        W2 = all_brains[:, b1_end:w2_end].reshape(COUNT, self.N_HID, self.N_OUT)
        b2 = all_brains[:, w2_end:b2_end].reshape(COUNT, self.N_OUT)
        return W1, b1, W2, b2

    def simulateImport(self, gen, startIndex, endIndex, fromCalmState):
        self.ensureGenBodies(gen) # checkpoint generations must be whole-gen loaded before the calm-state check below
        nodeCoor = self.getStartingNodeCoor(gen,startIndex,endIndex,fromCalmState)
        muscles = self.getMuscleArray(gen,startIndex,endIndex)
        brains_W1, brains_b1, brains_W2, brains_b2 = self.getBrainArrays(gen,startIndex,endIndex)
        currentFrame = 0
        return nodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2, currentFrame

    def simulateImportCreatures(self, creatures):
        """Trial import built from Creature OBJECTS instead of a generation row -
        works for pinned representatives whose generation row has been retired."""
        COUNT = len(creatures)
        nodeCoor = np.zeros((COUNT,self.CH+1,self.CW+1,self.node_coor_count))
        use_calm = any(c.calmState is not None for c in creatures)
        if not use_calm:
            # create grid of nodes along perfect gridlines
            coorGrid = np.mgrid[0:self.CW+1,0:self.CH+1]
            coorGrid = np.swapaxes(np.swapaxes(coorGrid,0,1),1,2)
            nodeCoor[:,:,:,0:2] = coorGrid
        else:
            for i,c in enumerate(creatures):
                if c.calmState is not None:
                    nodeCoor[i,:,:,:] = c.calmState
            nodeCoor[:,:,:,1] -= self.CH  # lift the creature above ground level
        muscles = self.buildMuscleArray([c.dna for c in creatures])
        brains_W1, brains_b1, brains_W2, brains_b2 = self.buildBrainArrays([c.dna for c in creatures])
        return nodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2, 0

    def ensureGenBodies(self, gen):
        """Loads a retired checkpoint generation's bodies back onto its shells.
        Whole-gen only: getStartingNodeCoor inspects shells[0].calmState to pick
        the starting pose for everyone."""
        if gen >= self.archived_upto: # hot generation - bodies are resident
            return
        if not self.store.has_snapshot(gen):
            return
        row = self.creatures[gen]
        if row is None:
            return
        if not self.body_cache.is_loaded(gen):
            self.body_cache.load(gen, row)

    def displayGenFor(self, gen):
        """Which generation's creatures to show when the slider sits on `gen`.
        Inside the hot window: itself. Older: the most recent checkpoint."""
        if gen >= self.archived_upto:
            return gen
        snap = (gen // self.snapshot_every) * self.snapshot_every
        return snap

    def loadCreatureDNA(self, creature):
        """Single-creature lazy reload used by rendering paths (icons/movies).
        Restores both dna and calmState so icons render the real pose."""
        gen = creature.IDNumber // self.c_count
        if not self.store.has_snapshot(gen):
            return # hot or dropped - nothing to load (dropped draws a placeholder)
        idx = creature.IDNumber % self.c_count
        entry = self.body_cache.peek(gen)
        if entry is None:
            row = self.creatures[gen]
            if row is None:
                return
            entry = self.body_cache.load(gen, row) # wires every shell of the gen
        if creature.dna is None:
            creature.dna = entry[0][idx]
        if creature.calmState is None and creature.dna is not None:
            creature.calmState = entry[1][idx].reshape(self.CH + 1, self.CW + 1, self.node_coor_count)

    def pinCreature(self, c):
        if c is None:
            return
        c.pinned = True
        self.pinned_creatures[c.IDNumber] = c

    def pinSpeciesReps(self, info):
        for ID in info.reps:
            if ID:
                self.pinCreature(self.getCreatureWithID(ID))

    def tempPinCreature(self, c):
        if c is not None:
            self.temp_pins.add(c)

    def clearTempPins(self):
        self.temp_pins.clear()

    def _checkArchiveDiskSpace(self):
        free = self.store.disk_free_bytes()
        if free < self.DISK_MIN_FREE_BYTES:
            raise RuntimeError(
                f"History archive drive nearly full ({free / 1024 ** 3:.1f} GB free). "
                f"Stopping before the archive is corrupted - free up space or delete "
                f"the '{self.store.dir}' folder to reclaim disk.")
        now = time.time()
        if free < self.DISK_WARN_FREE_BYTES and now - self._last_disk_warning_time > 3600:
            self._last_disk_warning_time = now
            print(f"[jes] WARNING: archive drive has {free / 1024 ** 3:.1f} GB free - "
                  f"consider deleting old history archives.")

    def _retireGeneration(self, g):
        """Called once per generation as it ages out of the hot window.
        Checkpoint generations spill their bodies to disk first; everything else
        just drops them. Pinned creatures keep their bodies in RAM either way."""
        row = self.creatures[g]
        if row is None:
            return
        is_checkpoint = (g % self.snapshot_every == 0)
        if is_checkpoint and not self.store.has_snapshot(g):
            self._checkArchiveDiskSpace()
            dna_block = np.stack([np.asarray(c.dna, dtype=np.float32) for c in row])
            calm_block = np.stack([np.asarray(c.calmState, dtype=np.float64) for c in row])
            self.store.write_snapshot(g, dna_block, calm_block)
        for c in row:
            if not self._isProtected(c):
                c.dna = None
                c.calmState = None
                c.icons = [None, None] # regenerable from dna; keeps checkpoint shells small
        if not is_checkpoint:
            self.creatures[g] = None # shells die too; pinned ones survive via pinned_creatures

    def frameToBeat(self, f):
        return (f//self.beat_time)%self.beats_per_cycle
        
    def frameToBeatFade(self, f):
        prog = f%self.beat_time
        return min(prog/self.beat_fade_time,1)

    def simulateRun(self, param, frameCount, calmingRun):
        nodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2, startCurrentFrame = param

        if HAS_NUMBA and jit_simulateRun is not None:
            newNodeCoor = jit_simulateRun(
                nodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2,
                startCurrentFrame, frameCount, calmingRun,
                self.beat_time, self.beats_per_cycle, self.gravity_acceleration_coef,
                self.calming_friction_coef, self.typical_friction_coef,
                self.ground_friction_coef, self.muscle_coef,
                self.y_clips[0], self.y_clips[1]
            )
            return newNodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2, startCurrentFrame + frameCount

        friction = self.calming_friction_coef if calmingRun else self.typical_friction_coef
        CEILING_Y = self.y_clips[0]
        FLOOR_Y = self.y_clips[1]
        
        nodeCoor = nodeCoor.copy()
        for f in range(frameCount):
            currentFrame = startCurrentFrame+f
            beat = 0
            
            if not calmingRun:
                beat = self.frameToBeat(currentFrame)
                nodeCoor[:,:,:,3] += self.gravity_acceleration_coef
                # decrease y-velo (3rd node coor) by G
            applyMuscles(nodeCoor,muscles[:,:,:,beat,:],self.muscle_coef)
            nodeCoor[:,:,:,2:4] *= friction
            nodeCoor[:,:,:,0:2] += nodeCoor[:,:,:,2:4]    # all node's x and y coordinates are adjusted by velocity_x and velocity_y
            
            if not calmingRun:    # dealing with collision with the ground.
                nodesTouchingGround = np.ma.masked_where(nodeCoor[:,:,:,1] >= FLOOR_Y, nodeCoor[:,:,:,1])
                m = nodesTouchingGround.mask.astype(float) # mask that only countains 1's where nodes touch the floor
                pressure = nodeCoor[:,:,:,1]-FLOOR_Y
                groundFrictionMultiplier = 0.5**(m*pressure*self.ground_friction_coef)
                
                nodeCoor[:,:,:,1] = np.clip(nodeCoor[:,:,:,1], CEILING_Y, FLOOR_Y) # clip nodes below the ground back to ground level
                nodeCoor[:,:,:,2] *= groundFrictionMultiplier # any nodes touching the ground must be slowed down by ground friction.
        
        if calmingRun: # If it's a calming run, then take the average location of all nodes to center it at the origin.
            nodeCoor[:,:,:,0] -= np.mean(nodeCoor[:,:,:,0], axis=(1,2), keepdims=True)
        return nodeCoor, muscles, brains_W1, brains_b1, brains_W2, brains_b2, startCurrentFrame+frameCount  
        
    def doSpeciesInfo(self,nsp,best_of_each_species,gen):
        nsp = dict(sorted(nsp.items()))
        running = 0
        alive_now = set()
        if len(nsp) > 0:
            top_sp = max(nsp, key=lambda k: nsp[k][0])
            if top_sp == self.current_leader:
                self.leader_tenure += 1
            else:
                self.current_leader = top_sp
                self.leader_tenure = 1

        for sp in nsp.keys():
            pop = nsp[sp][0]
            nsp[sp][1] = running
            nsp[sp][2] = running+pop
            running += pop
            alive_now.add(sp)

            info = self.species_info[sp]
            info.last_seen_gen = gen
            new_rep = best_of_each_species[sp]
            info.reps[3] = new_rep # most-recent representative (hot window covers it while alive)
            if pop > info.apex_pop: # This species reached its highest population
                info.apex_pop = pop
                info.reps[2] = new_rep # apex representative
                if info.prominent:
                    self.pinCreature(self.getCreatureWithID(new_rep))
            if pop >= self.c_count*self.S_NOTABLE and not info.prominent:  #prominent threshold
                info.becomeProminent()

            # Record fitness for adaptive incubation tracking
            best_creature = self.getCreatureWithID(best_of_each_species[sp])
            info.record_fitness(gen, best_creature.fitness)

        # Species that just went extinct lose hot-window coverage: freeze the
        # replayable lineage of any species a genealogy hover / S-store can reach.
        for sp in self._alive_last_gen - alive_now:
            info = self.species_info[sp]
            if info.prominent or (self.ui is not None and sp == getattr(self.ui, "species_storage", None)):
                self.pinSpeciesReps(info)
        self._alive_last_gen = alive_now
                
    def checkALAP(self):
        if self.ui.ALAPButton.setting == 1: # We're already ALAP-ing!
            self.doGeneration(self.ui.doGenButton)
        
    def doGeneration(self, button):
        generation_start_time = time.time() #calculates how long each generation takes to run
        
        gen = len(self.creatures)-1
        creatureState = self.simulateImport(gen, 0, self.c_count, True)
        simResult = self.simulateRun(creatureState, self.trial_time, False)
        nodeCoor = simResult[0]
        rawScores = nodeCoor[:,:,:,0].mean(axis=(1, 2)) # find each creature's average X-coordinate
        finalScores = np.abs(rawScores) # Absolute speed / displacement
        
        # DNA Mirroring: for any creature moving left, horizontally flip its DNA & calmState
        # so that its revolutionary locomotion physics are rescued and propelled to the right
        for c in range(self.c_count):
            if rawScores[c] < 0:
                self.creatures[gen][c].dna = flipDNA(self.creatures[gen][c].dna, self.CW, self.CH, self.beats_per_cycle, self.traits_per_box)
                if self.creatures[gen][c].calmState is not None:
                    self.creatures[gen][c].calmState[:, :, 0] = -np.flip(self.creatures[gen][c].calmState[:, :, 0], axis=1)
                    self.creatures[gen][c].calmState[:, :, 1] = np.flip(self.creatures[gen][c].calmState[:, :, 1], axis=1)

        # Tallying up all the data
        currRankings = np.flip(np.argsort(finalScores),axis=0)
        newPercentiles = np.zeros((self.HUNDRED+1))
        newSpeciesPops = {}
        best_of_each_species = {}
        for rank in range(self.c_count):
            c = currRankings[rank]
            self.creatures[gen][c].fitness = finalScores[c]
            self.creatures[gen][c].rank = rank
            
            species = self.creatures[gen][c].species
            if species in newSpeciesPops:
                newSpeciesPops[species][0] += 1
            else:
                newSpeciesPops[species] = [1,None,None]
            if species not in best_of_each_species:
                best_of_each_species[species] = self.creatures[gen][c].IDNumber
        self.doSpeciesInfo(newSpeciesPops,best_of_each_species,gen)

        for p in range(self.HUNDRED+1):
            rank = min(int(self.c_count*p/self.HUNDRED),self.c_count-1)
            c = currRankings[rank]
            newPercentiles[p] = self.creatures[gen][c].fitness
        
        # Tiered merit-based incubation protection:
        pop_median = newPercentiles[50]
        pop_max = newPercentiles[0]
        incubating_species = {sp for sp in newSpeciesPops.keys() if self.species_info[sp].is_incubating(gen, self.leader_tenure, pop_median, pop_max)}
        
        # Sort candidate incubating species by performance so top newcomers get first priority
        incubating_ranked = sorted(list(incubating_species), key=lambda sp: self.species_info[sp].best_fitness, reverse=True)
        
        # Max 15% quota of total population for incubation protection
        max_incubator_slots = max(2, int(self.c_count * 0.15))
        protected_creatures = set()
        protected_slots_used = 0
        species_representatives = {}
        
        for sp in incubating_ranked:
            if protected_slots_used >= max_incubator_slots:
                break
            for rank in range(self.c_count):
                c = currRankings[rank]
                if self.creatures[gen][c].species == sp:
                    reps = species_representatives.setdefault(sp, [])
                    if len(reps) < self.min_incubation_survivors:
                        reps.append(c)
                        protected_creatures.add(c)
                        protected_slots_used += 1
                        if protected_slots_used >= max_incubator_slots:
                            break
        
        # Option 2: Anti-monopoly carrying capacity cap (max 50% population per species)
        max_cap = max(2, int(self.c_count * self.max_species_fraction))
        species_offspring_count = {sp: 0 for sp in newSpeciesPops}
        WILD_EXILE_PROB = 0.10 # Calibrated speciation rate: prevents TV static and keeps median climbing

        currCreatures = self.creatures[-1]
        nextCreatures = [None]*self.c_count
        for rank in range(self.c_count//2):
            winner = currRankings[rank]
            loser = currRankings[(self.c_count-1)-rank]
            if random.uniform(0,1) < rank/self.c_count:
                ph = loser
                loser = winner
                winner = ph
            
            w_species = self.creatures[gen][winner].species
            nextCreatures[winner] = None
            
            # Winner reproduction with calibrated carrying capacity cap
            over_cap_w = (species_offspring_count.get(w_species, 0) >= max_cap)
            if over_cap_w:
                if random.uniform(0, 1) < WILD_EXILE_PROB:
                    nextCreatures[winner] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+winner, force_big_mutation=True)
                else:
                    nextCreatures[winner] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+winner)
            elif random.uniform(0,1) < rank/self.c_count*2.0:
                nextCreatures[winner] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+winner)
                species_offspring_count[w_species] = species_offspring_count.get(w_species, 0) + 1
            else:
                nextCreatures[winner] = self.clone(self.creatures[gen][winner], (gen+1)*self.c_count+winner)
                species_offspring_count[w_species] = species_offspring_count.get(w_species, 0) + 1
            
            # Loser reproduction: check innovation incubation protection
            if loser in protected_creatures:
                # Young species innovation protection: breed within its own lineage
                l_creature = self.creatures[gen][loser]
                nextCreatures[loser] = self.mutate(l_creature, (gen+1)*self.c_count+loser)
                species_offspring_count[l_creature.species] = species_offspring_count.get(l_creature.species, 0) + 1
                self.creatures[gen][loser].living = True
            else:
                # Standard loser replacement
                over_cap_l = (species_offspring_count.get(w_species, 0) >= max_cap)
                if over_cap_l:
                    if random.uniform(0, 1) < WILD_EXILE_PROB:
                        nextCreatures[loser] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+loser, force_big_mutation=True)
                    else:
                        nextCreatures[loser] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+loser)
                else:
                    nextCreatures[loser] = self.mutate(self.creatures[gen][winner], (gen+1)*self.c_count+loser)
                    species_offspring_count[w_species] = species_offspring_count.get(w_species, 0) + 1
                self.creatures[gen][loser].living = False
        
        self.creatures.append(nextCreatures)
        self.rankings.append(currRankings.astype(np.int32))
        self.percentiles.append(newPercentiles.astype(np.float32))
        self.species_pops.append(newSpeciesPops)
        
        drawAllGraphs(self, self.ui)
        
        self.getCalmStates(gen+1,0,self.c_count,self.stabilization_time,True)
        #Calm the creatures down so no potential energy is stored
  
        self.ui.genSlider.val_max = gen+1
        self.ui.genSlider.manualUpdate(gen)
        self.last_gen_run_time = time.time()-generation_start_time

        # Tiered history maintenance: as generations age out of the hot window,
        # checkpoint gens spill their bodies to disk and everything else drops them.
        target = (gen + 1) - self.ram_window_gens
        while 0 <= self.archived_upto <= target:
            self._retireGeneration(self.archived_upto)
            self.archived_upto += 1

        self.ui.detectMouseMotion()
        
    def getCreatureWithID(self, ID):
        row = self.creatures[ID // self.c_count]
        if row is not None:
            return row[ID % self.c_count]
        return self.pinned_creatures.get(ID) # retired non-checkpoint gen: pinned reps only
        
    def clone(self, parent, newID):
        return Creature(parent.dna, newID, parent.species, self, self.ui)
        
    def mutate(self, parent, newID, force_big_mutation=False):
        newDNA, newSpecies, cwc = parent.getMutatedDNA(self, force_big_mutation=force_big_mutation)
        newCreature = Creature(newDNA, newID, newSpecies, self, self.ui)
        if newCreature.species != parent.species:
            self.species_info.append(SpeciesInfo(self,newCreature,parent))
            newCreature.codonWithChange = cwc
        return newCreature