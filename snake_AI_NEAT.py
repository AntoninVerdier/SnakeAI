import curses
import random
import math
from curses import textpad
import numpy as np
import time
import argparse
import pickle
import pandas as pd 
from tqdm import tqdm
import os
import neat

parser = argparse.ArgumentParser(description='AI plays snake')

parser.add_argument('--screen_play', '-sp', action='store_true',
					 help='Display the game, do not use for training')

args = parser.parse_args()

key_to_press = [curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN, curses.KEY_UP]

reverse = [[curses.KEY_RIGHT, curses.KEY_LEFT],
		   [curses.KEY_LEFT, curses.KEY_RIGHT],
		   [curses.KEY_UP, curses.KEY_DOWN],
		   [curses.KEY_DOWN,  curses.KEY_UP]]


def create_food(snake, box):
	food = None
	while food is None:
		food = [random.randint(box[0][0] +1, box[1][0] -1), 
				random.randint(box[0][1] +1, box[1][1] -1)]
		if food in snake:
			food = None

	return food

def init_snake(sh, sw):
	snake = [[sh//2, sw//2 +1],[sh//2, sw//2],[sh//2, sw//2 -1]]
	direction = curses.KEY_RIGHT
	moves = 200
	lifetime = 0
	score = 0

	return snake, direction, moves, lifetime, score

def init_screen(stdscr):
	curses.curs_set(0)
	stdscr.nodelay(1)
	stdscr.timeout(80)
	# sh, sw = stdscr.getmaxyx()
	sh, sw = 54, 54
	box = [[7, 7], [sh-7, sw-7]]
	textpad.rectangle(stdscr, box[0][0], box[0][1], box[1][0], box[1][1])
	return sh, sw, box

def distances_from_border(box, snake):
	dist_border_up = snake[0][0] - box[0][0]
	dist_border_down = box[1][0] - snake[0][0]
	dist_border_right = box[1][1] - snake[0][1]
	dist_border_left= snake[0][1] - box[0][1]

	if dist_border_up > dist_border_left:
		dist_border_up_left = math.sqrt(2)*dist_border_left
	else:
		dist_border_up_left = math.sqrt(2)*dist_border_up

	if dist_border_up > dist_border_right:
		dist_border_up_right = math.sqrt(2)*dist_border_right
	else:
		dist_border_up_right = math.sqrt(2)*dist_border_up

	if dist_border_down > dist_border_left:
		dist_border_down_left = math.sqrt(2)*dist_border_left
	else:
		dist_border_down_left = math.sqrt(2)*dist_border_down

	if dist_border_down > dist_border_right:
		dist_border_down_right = math.sqrt(2)*dist_border_right
	else:
		dist_border_down_right = math.sqrt(2)*dist_border_down

	dist_directions = [dist_border_up/40,  dist_border_down/40, 
					   dist_border_right/40, dist_border_left/40, 
					   dist_border_up_left/40, dist_border_up_right/40,
					   dist_border_down_left/40, dist_border_down_right/40]
	
	return dist_directions

def food_direction(food, snake):
	food_up, food_down, food_right, food_left = 0, 0, 0, 0
	food_up_right, food_up_left, food_down_right, food_down_left = 0, 0, 0, 0

	# Get if food is on the direction
	if food[0] == snake[0][0]:
		if snake[0][1] > food[1]:
			food_left = 1
		elif snake[0][1] < food[1]:
			food_right = 1

	if food[1] == snake[0][1]:
		if snake[0][0] > food[0]:
			food_up = 1
		elif snake[0][0] < food[0]:
			food_down = 1

	if abs(snake[0][0] - food[0]) == abs(snake[0][1] - food[1]):
		if food[0] > snake[0][0] and food[1] > snake[0][1]:
			food_down_right = 1
		elif food[0] < snake[0][0] and food[1] > snake[0][1]:
			food_up_right = 1
		elif food[0] > snake[0][0] and food[1] < snake[0][1]:
			food_down_left = 1
		elif food[0] < snake[0][0] and food[1] < snake[0][1]:
			food_up_left = 1

	food_on_path = [food_up, food_down, food_right, food_left, 
					food_up_right, food_up_left, food_down_right, food_down_left]

	return food_on_path

def snake_collision(snake):
	snake_up, snake_down, snake_right, snake_left = 0, 0, 0, 0
	snake_up_right, snake_up_left, snake_down_right, snake_down_left = 0, 0, 0, 0
	
	for parts in snake[1:]:
		if snake[0][0] == parts[0] and snake[0][1] > parts[1]:
			snake_left = 1
		elif snake[0][0] == parts[0] and snake[0][1] < parts[1]:
			snake_right = 1
		elif snake[0][1] == parts[1] and snake[0][0] > parts[0]:
			snake_up = 1
		elif snake[0][1] == parts[1] and snake[0][0] < parts[0]:
			snake_down = 1
		if abs(snake[0][0] - parts[0]) == abs(snake[0][1] - parts[1]):
			if parts[0] > snake[0][0] and parts[1] > snake[0][1]:
				snake_down_right = 1
			elif parts[0] < snake[0][0] and parts[1] > snake[0][1]:
				snake_up_right = 1
			elif parts[0] > snake[0][0] and parts[1] < snake[0][1]:
				snake_down_left = 1
			elif parts[0] < snake[0][0] and parts[1] < snake[0][1]:
					snake_up_left = 1



		snake_collisions = [snake_up, snake_down, snake_right, snake_left,
							snake_up_right, snake_up_left, snake_down_right, snake_down_left]

	return snake_collisions

def end_game(snake, moves, box):
	if snake[0][0] in [box[0][0], box[1][0]] or \
	   snake[0][1] in [box[0][1], box[1][1]] or \
	   snake[0] in snake[1:] or moves == 0:

	   return True
	return False

def get_fitness(lifetime, score):
	if score < 10:
		fitness = lifetime*(2**score)
	else:
		fitness = math.floor(lifetime)*(2**score)*(score - 9)

	return fitness

def eval_genomes(genomes, config):	
	sh, sw = 54, 54
	box = [[7, 7], [sh-7, sw-7]]

	for genome_id, genome in genomes:
		genome.fitness = 0.0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		#for idx, model in enumerate(current_pool):

		snake, direction, moves, lifetime, score = init_snake(sh, sw)
		food = create_food(snake, box)
		while True:
			lifetime += 1

			dist_directions = distances_from_border(box, snake)
			food_on_path = food_direction(food, snake)
			snake_collisions = snake_collision(snake)

			input_data = np.array(dist_directions + food_on_path + snake_collisions)
			prediction = net.activate(input_data)
			key = key_to_press[np.array(prediction).argmax()]

			for i in reverse:
				if key == i[0] and direction != i[1]:
					direction = key

			head = snake[0]

			if direction == curses.KEY_RIGHT:
				new_head = [head[0], head[1] +1]
			elif direction == curses.KEY_LEFT:
				new_head = [head[0], head[1] -1]
			elif direction == curses.KEY_UP:
				new_head = [head[0] -1, head[1]]
			elif direction == curses.KEY_DOWN:
				new_head = [head[0] +1, head[1]]
			
			snake.insert(0, new_head)

			if food in snake:
				if moves + 100 <= 500:
					moves = moves + 100
				food = create_food(snake, box)
				score += 1
			else:
				moves -= 1
				snake.pop()

			fitness = get_fitness(lifetime, score)

			### Check if endgame

			if end_game(snake, moves, box):
				genome.fitness = fitness
				break


def replay(stdscr):
	sh, sw, box = init_screen(stdscr)
	print(sw, sh, box)
	snake, direction, moves, lifetime, score = init_snake(sh, sw)
	food = create_food(snake, box)
	# with open(os.path.join('best_snake', 'pickle_snake.pkl'), 'rb') as f:
	# 	model = pickle.load(f)
	stdscr.addstr(sh-1, 0, str(sh) + ',' + str(sw))

	for y, x in snake:
		stdscr.addstr(y, x, '#')
	stdscr.addstr(food[0], food[1], '*')
	stdscr.addstr(0, sw//2, 'Score: {}'.format(score))

	while True:
		lifetime += 1

		dist_directions = distances_from_border(box, snake)
		food_on_path = food_direction(food, snake)
		snake_collisions = snake_collision(snake)

		input_data = np.array(dist_directions + food_on_path + snake_collisions)
		prediction =  winner_net.activate(input_data)
		key = key_to_press[np.array(prediction).argmax()]

		for i in reverse:
			if key == i[0] and direction != i[1]:
				direction = key

		#### FOR HUMAN PLAY ####
		#key = stdscr.getch()
		# if key in [curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN, curses.KEY_UP]:
			# for i in reverse:	
			# 	if key == i[0] and direction != i[1]:
			# 		direction = key


		head = snake[0]

		if direction == curses.KEY_RIGHT:
			new_head = [head[0], head[1] +1]
		elif direction == curses.KEY_LEFT:
			new_head = [head[0], head[1] -1]
		elif direction == curses.KEY_UP:
			new_head = [head[0] -1, head[1]]
		elif direction == curses.KEY_DOWN:
			new_head = [head[0] +1, head[1]]
		
		snake.insert(0, new_head)
		stdscr.addstr(new_head[0], new_head[1], '#')

		if food in snake:
			if moves + 100 <= 500:
				moves = moves + 100
			food = create_food(snake, box)
			stdscr.addstr(food[0], food[1], '*')
			score += 1
		else:
			moves -= 1
			stdscr.addstr(snake[-1][0], snake[-1][1], ' ')
			snake.pop()

		fitness = get_fitness(lifetime, score)


		### Display informations


		for i, dist_direction in enumerate(['Up', 'Down', 'Right', 'Left', 'haut_gauche']):
			stdscr.addstr(1, 15*i, (dist_direction + ': {:.02f}').format(dist_directions[i]))

		for i, dist_direction in enumerate(['Haut_gauche', 'Haut_droit', 'Bas_gauche', 'Bas_droit']):
			stdscr.addstr(2, 20*i, (dist_direction + ': {:.02f}').format(dist_directions[i+4]))

		for i, dire in enumerate(['Up', 'Down', 'Right', 'Left']):
			stdscr.addstr(3, 15*i, (dire + ': {}').format(food_on_path[i]))
		
		for i, dire in enumerate(['Up_right', 'Up_left', 'Down_right', 'Down_left']):
			stdscr.addstr(4, 15*i, (dire + ': {}').format(food_on_path[i+4]))

		for i, dire in enumerate(['Up_collide', 'Down_collide', 'Right_collide', 'Left_collide']):
			stdscr.addstr(5, 20*i, (dire + ': {}').format(snake_collisions[i]))
		
		for i, dire in enumerate(['Up_right_collide', 'Up_left_collide', 'Down_right_collide', 'Down_left_collide']):
			stdscr.addstr(6, 25*i, (dire + ': {}').format(snake_collisions[i+4]))

		stdscr.addstr(0, 0, 'NN_key : {}'.format(key))
		stdscr.addstr(0, sw//4, 'Moves left : {:03d}'.format(moves))
		stdscr.addstr(0, sw//2, 'Score: {}'.format(score))
		stdscr.addstr(0, sw*3//4, 'Fitness : {:d}'.format(fitness))

		### Check if endgame

		if end_game(snake, moves, box):
			msg = 'Game Over !'
			stdscr.addstr(sh//2, sw//2 - len(msg)//2, msg)
			stdscr.nodelay(0)
			stdscr.getch()
			break
		
		time.sleep(0.05)
		stdscr.refresh()

def main(stdscr):
 	replay(stdscr, config, winner_net)


if args.screen_play:
	curses.wrapper(main)
else:
	config_file = os.path.join('config_NEAT.py')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_file)

# 	# Create the population, which is the top-level object for a NEAT run.
# 	p = neat.Population(config)

# 	#Add a stdout reporter to show progress in the terminal.
# 	p.add_reporter(neat.StdOutReporter(True))
# 	stats = neat.StatisticsReporter()
# 	p.add_reporter(stats)
# 	p.add_reporter(neat.Checkpointer(5))

# 	# Run for up to 300 generations.
# 	winner = p.run(eval_genomes, 300)

	# pickle.dump(winner, open(os.path.join('best_snake', 'pickle_snake.pkl'), 'wb'))

	with open(os.path.join('best_snake', 'pickle_snake.pkl'), 'rb') as f:
		winner = pickle.load(f)
	
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
	curses.wrapper(replay)




