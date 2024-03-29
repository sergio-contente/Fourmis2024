"""
Module managing an ant colony in a labyrinth.
"""
import numpy as np
import maze
import pheromone
import direction as d
import pygame as pg

UNLOADED, LOADED = False, True

exploration_coefs = 0.


class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, start_nb, end_nb, pos_init, max_life):
        # Each ant has is own unique random seed
        self.seeds = np.arange(start_nb+1, end_nb+1, dtype=np.int64)
        # State of each ant : loaded or unloaded
        self.is_loaded = np.zeros(end_nb-start_nb, dtype=np.int8)
        # Compute the maximal life amount for each ant :
        #   Updating the random seed :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Amount of life for each ant = 75% à 100% of maximal ants life
        self.max_life = max_life * np.ones(end_nb-start_nb, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Ages of ants : zero at beginning
        self.age = np.zeros(end_nb-start_nb, dtype=np.int64)
        # History of the path taken by each ant. The position at the ant's age represents its current position.
        self.historic_path = np.zeros((end_nb-start_nb, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction in which the ant is currently facing (depends on the direction it came from).
        self.directions = d.DIR_NONE*np.ones(end_nb-start_nb, dtype=np.int8)
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Function that returns the ants carrying food to their nests.

        Inputs :
            loaded_ants: Indices of ants carrying food
            pos_nest: Position of the nest where ants should go
            food_counter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loaded_ants] -= 1

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0
                food_counter += in_nest_loc.shape[0]
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for
                          easier edge management)

        Outputs: None
        """
        # Update of the random seed (for manual pseudo-random) applied to all unloaded ants
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calculating possible exits for each ant in the maze:
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        # Reading neighboring pheromones:
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calculating choices for all ants not carrying food (for others, we calculate but it doesn't matter)
        choices = self.seeds[:] / 2147483647.

        # Ants explore the maze by choice or if no pheromone can guide them:
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calculating indices of ants whose last move was not valid:
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choosing a random direction:
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Valid move if we didn't stay in place due to a wall
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # and if we're not in the opposite direction of the previous move (and if there are other exits)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calculating indices of ants whose move we just validated:
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # For these ants, we update their positions and directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Aging one unit for the age of ants not carrying food
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Killing ants at the end of their life:
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # For ants reaching food, we update their states:
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0
        # Marking pheromones:
        old_pheromones = pheromones.pheromon.copy()
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]], old_pheromon = old_pheromones ) for i in range(self.directions.shape[0])]
        return food_counter

    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]


if __name__ == "__main__":
    import sys
    import time
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size() 

    color = 0 if rank == 0 else 1

    comm_calc = comm.Split(color)
    rank_calc = comm_calc.Get_rank()
    size_calc = comm_calc.Get_size()


    size_laby = 25, 25
    if len(sys.argv) > 2:
        size_laby = int(sys.argv[1]),int(sys.argv[2])

    
    pg.init()
    resolution = size_laby[1]*8, size_laby[0]*8
    if rank==0:
        screen = pg.display.set_mode(resolution)
        temps_total = 0
    else:
        screen = pg.display.set_mode(resolution, flags=pg.HIDDEN)
    
    nb_ants = size_laby[0]*size_laby[1]//4
    max_life = 1000 # Todo processo tem o max_life
    if len(sys.argv) > 3:
        max_life = int(sys.argv[3])
    pos_food = size_laby[0]-1, size_laby[1]-1 #Todo processo recebe o pos_food
    pos_nest = 0, 0 # Todo processo recebe o pos-nest
    alpha = 0.9
    beta  = 0.99

    a_maze = maze.Maze(size_laby, 12345) # Todo processo tem o mesmo maze

    if rank == 0:
        ants_global = Colony(0, nb_ants, pos_nest, max_life)
        pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
   

    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])
    food_counter = 0


    if color != 0:
        # Calcualitng Nloc for all process involved in calculation
        rest = nb_ants % (size_calc)
        Nloc = (nb_ants // (size_calc)) + (1 if rank_calc <= rest and rank_calc != 0 else 0)

        # Preparing arrays to represent the distribution among processes
        recv_count = np.empty(size_calc, dtype=np.int32)
        comm_calc.Allgather(np.array([Nloc], dtype=np.int32), recv_count)
        displacements = np.insert(np.cumsum(recv_count[:-1]), 0, 0)

        historic_recvcounts = recv_count.copy() * (max_life+1) * 2
        historic_displacements = displacements.copy() * (max_life+1) * 2

        # Now, we can define start and end indexes for each process
        block_start = displacements[rank_calc]
        block_end = block_start + Nloc

        # Local objects
        ants_local = Colony(block_start, block_end, pos_nest, max_life)
        food_counter_local = np.array(0, dtype=np.int64)
        pherom_local = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    
        # Buffers initialization
        if rank_calc == 0:
            seeds_colored = np.empty(nb_ants, dtype=np.int64)
            is_loaded_colored = np.empty(nb_ants, dtype=np.int8)
            age_colored = np.empty(nb_ants, dtype=np.int64)
            historic_path_colored = np.empty((nb_ants, max_life+1, 2), dtype=np.int16)
            directions_colored = np.empty(nb_ants, dtype=np.int8)
            food_counter_colored = np.empty(1, np.int64)
            pheromon_colored = np.empty((size_laby[0]+2, size_laby[1]+2), dtype=np.double)
            
        else:
            seeds_colored = None
            is_loaded_colored = None
            age_colored = None
            historic_path_colored = None
            directions_colored = None
            food_counter_colored = None
            pheromon_colored = None


    for cycle in range(0, 7000):
        deb = time.time()
        if color != 0:
            # Calculation processes
            food_counter_local = np.array(ants_local.advance(a_maze, pos_food, \
                pos_nest, pherom_local, food_counter_local), dtype=np.int64)
            pherom_local.do_evaporation(pos_food)
            pherom_send = np.array(pherom_local.pheromon.copy())


            comm_calc.Reduce([food_counter_local, MPI.INT64_T], \
                              [food_counter_colored, MPI.INT64_T], op=MPI.SUM, root=0)
            comm_calc.Reduce([pherom_send, MPI.DOUBLE], [pheromon_colored, MPI.DOUBLE],\
                             op=MPI.MAX, root=0)
            pherom_local.pheromon = comm_calc.bcast(pheromon_colored, root=0)
            comm_calc.Gatherv(ants_local.age, \
                              [age_colored, recv_count, displacements, MPI.INT64_T], root=0)
            comm_calc.Gatherv(ants_local.historic_path, \
                              [historic_path_colored, historic_recvcounts,\
                                historic_displacements, MPI.INT16_T], root=0)
            comm_calc.Gatherv(ants_local.directions, \
                              [directions_colored, recv_count, displacements, MPI.INT8_T], root=0)
            comm.send([seeds_colored, is_loaded_colored, age_colored,\
                        historic_path_colored, directions_colored, \
                            food_counter_colored, pheromon_colored], dest=0)
        else:
            mazeImg = a_maze.display()        
            ants_attributes = comm.recv(source=1)
            # Updating ants
            ants_global.seeds = ants_attributes[0]
            ants_global.is_loaded = ants_attributes[1]
            ants_global.age = ants_attributes[2]
            ants_global.historic_path = ants_attributes[3]
            ants_global.directions = ants_attributes[4]
            food_counter = ants_attributes[5]
            pherom.pheromon = ants_attributes[6]
            ## Print screen pygame window
            snapshop_taken = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit(0)

            if food_counter[0] == 1 and not snapshop_taken:
                pg.image.save(screen, "MyFirstFood.png")
                snapshop_taken = True   
            
            # Updating display
            pherom.display(screen)
            screen.blit(mazeImg, (0, 0))
            ants_global.display(screen)
            pg.display.update()
            end = time.time()
            temps_total += end - deb
            print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter[0]:7d}")

    if rank==0:
        print(f"Temps total: {temps_total}")
        output_str = f"Temps total: {temps_total}"
        # Open the file in append mode ('a') to add to the file without overwriting it
        with open('results_parallelized_Q2.txt', 'w') as file:
            file.write(output_str + '\n')  # Write the output string to the file, adding a newline character at the end

        
