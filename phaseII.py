from enum import Enum
from hitman_referee import *

from itertools import product,combinations
from typing import List, Tuple, Dict , Set, Any
import sys
import subprocess
from pprint import *
import random as rd
from collections import namedtuple
from collections import deque
from collections import defaultdict
import heapq
import copy

import numpy as np

Grid = List[List[int]]
ACTIONS =[""]

class World(Enum):
    EMPTY = 1
    WALL = 2
    GUARD_N = 3
    GUARD_E = 4
    GUARD_S = 5
    GUARD_W = 6
    CIVIL_N = 7
    CIVIL_E = 8
    CIVIL_S = 9
    CIVIL_W = 10
    TARGET = 11
    SUIT = 12
    PIANO_WIRE = 13

def grid_to_coords_dict(grid: Grid) -> dict:
    """
    :param grid: grid of the level
    :return: dict containing position of each element
    """
    coords = {
        "cells": [],
        "empty": [],
        "hero": [],
        "guards": [],
        "targets": [],
        "walls": [],
        "costumes": [],
        "ropes": [],
        "nothing": [],
    }

    for i, line in enumerate(grid):
        for j, cell in enumerate(line):
            if cell != "#":
                coords["cells"].append((i, j))
            if cell in [" ", "G", "T", "W", "C", "R", "N"]:
                coords["empty"].append((i, j))
            if cell == "H":
                coords["hero"].append((i, j))
            elif cell == "G":
                coords["guards"].append((i, j))
            elif cell == "T":
                coords["targets"].append((i, j))
            elif cell == "W":
                coords["walls"].append((i, j))
            elif cell == "C":
                coords["costumes"].append((i, j))
            elif cell == "R":
                coords["ropes"].append((i, j))
            elif cell == "N":
                coords["nothing"].append((i, j))

    return coords


def vocabulary(coords: dict) -> dict:
    """
    :param coords: dict containing coord of each element of the map
    :return: dict containing all the vocabulary
    """
    cells = coords["cells"]
    targets = coords["targets"]

    act_vars = [("do", a) for a in ACTIONS]
    at_vars = [("at", c) for c in cells]
    vision_vars = [("vision", c) for c in cells]
    hear_vars = [("hear", c) for c in cells]
    hero_vars = [("hero", c) for c in cells]
    guard_vars = [("guard", c) for c in cells]
    target_vars = [("target", c) for c in targets]
    wall_vars = [("wall", c) for c in cells]
    costume_vars = [("costume", c) for c in cells]
    rope_vars = [("rope", c) for c in cells]
    nothing_vars = [("nothing", c) for c in cells]

    return {
        v: i + 1
        for i, v in enumerate(
            act_vars
            + at_vars
            + vision_vars
            + hear_vars
            + hero_vars
            + guard_vars
            + target_vars
            + wall_vars
            + costume_vars
            + rope_vars
            + nothing_vars
        )
    }



def is_found_suit(state) -> bool:
    return state['has_suit'] is True


def is_found_weapon(state) -> bool:
    return state['has_weapon'] is True


def is_suit_worn(state) -> bool:
    return state['is_suit_on'] is True


def is_killed_target(state) -> bool:
    return state['is_down_target'] is True


def is_seen_by_guard(state) -> bool:
    return state['is_in_guard_range'] is True


def is_seen_by_civil(state) -> bool:
    return state['is_in_civil_range'] is True


def is_terminal(state) -> bool:
    return state['position'] == (0, 0) and state['is_target_down'] is True


def is_valid_position(state, x: int, y: int, world_example: List) -> bool:
    # Vérifiez les limites de la grille
    if x < 0 or x >= state['m'] or y < 0 or y >= state['n']:
        return False

    # Vérifiez les obstacles (murs, gardes, etc.)
    if world_example[x][y] not in [
        HC.EMPTY,
        HC.PIANO_WIRE,
        HC.CIVIL_N,
        HC.CIVIL_E,
        HC.CIVIL_S,
        HC.CIVIL_W,
        HC.SUIT,
        HC.TARGET,
    ]:
        return False

    return True


# BFS
def get_path_to_position_bfs(referee: HitmanReferee(), state, start_position: Tuple[int, int], target_position: Tuple[int, int], world_example):
    # Utilisez l'algorithme BFS pour trouver le chemin le plus court
    queue = deque()
    queue.append(start_position)

    # Utilisez un dictionnaire pour enregistrer les positions parentes pour reconstruire le chemin
    parents = {start_position: None}

    while queue:
        current_position = queue.popleft()

        if current_position == target_position:
            # Le chemin a été trouvé ! Reconstruire le chemin parcouru
            path = []
            while current_position is not None:
                path.append(current_position)
                current_position = parents[current_position]
            return list(reversed(path))

        x, y = current_position

        # Générer les positions voisines valides et les ajouter à la file d'attente
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if is_valid_position(state, nx, ny, world_example) and (nx, ny) not in parents:
                queue.append((nx, ny))
                parents[(nx, ny)] = current_position

    # Aucun chemin trouvé
    return None


# DFS
def get_path_to_position_dfs(referee, state, start_position, target_position, world_example):
    # Utiliser l'algorithme DFS pour trouver le chemin
    stack = []
    stack.append(start_position)

    # Utiliser un dictionnaire pour enregistrer les positions parentes pour reconstruire le chemin
    parents = {start_position: None}

    while stack:
        current_position = stack.pop()

        if current_position == target_position:
            # Le chemin a été trouvé ! Reconstruire le chemin parcouru
            path = []
            while current_position is not None:
                path.append(current_position)
                current_position = parents[current_position]
            return list(reversed(path))

        x, y = current_position

        # Générer les positions voisines valides et les ajouter à la pile
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if is_valid_position(state, nx, ny, world_example) and (nx, ny) not in parents:
                stack.append((nx, ny))
                parents[(nx, ny)] = current_position

    # Aucun chemin trouvé
    return None


# IDDFS
def get_path_to_position_iddfs(referee: HitmanReferee(), state, start_position, target_position, world_example):
    # Utiliser l'algorithme IDDFS pour trouver le chemin
    depth = 0
    while True:
        result = depth_limited_dfs(
            referee, state, start_position, target_position, depth, world_example)
        if result is not None:
            return result
        depth += 1


def depth_limited_dfs(referee: HitmanReferee(), state, start_position, target_position, depth_limit, world_example):
    stack = [(start_position, 0)]

    # Utiliser un dictionnaire pour enregistrer les positions parentes pour reconstruire le chemin
    parents = {start_position: None}

    while stack:
        current_position, current_depth = stack.pop()

        if current_position == target_position:
            # Le chemin a été trouvé ! Reconstruire le chemin parcouru
            path = []
            while current_position is not None:
                path.append(current_position)
                current_position = parents[current_position]
            return list(reversed(path))

        if current_depth < depth_limit:
            x, y = current_position

            # Générer les positions voisines valides et les ajouter à la pile
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if is_valid_position(state, nx, ny, world_example) and (nx, ny) not in parents:
                    stack.append(((nx, ny), current_depth + 1))
                    parents[(nx, ny)] = current_position

    # Aucun chemin trouvé dans la limite de profondeur donnée
    return None


# A*
def heuristic_manhattan(position, target_position):
    # Heuristique : distance de Manhattan
    x1, y1 = position
    x2, y2 = target_position
    return abs(x2 - x1) + abs(y2 - y1)


action_count = 0
seen_by_guards_count = 0
neutralized_count = 0
seen_in_costume_count = 0
seen_neutralizing_count = 0
seen_killing_target_count = 0


def get_action_count() -> int:
    global action_count
    action_count += 1
    return action_count


def get_seen_by_guards_count() -> int:
    global seen_by_guards_count
    seen_by_guards_count += 1
    return seen_by_guards_count


def get_neutralized_count() -> int:
    global neutralized_count
    neutralized_count += 1
    return neutralized_count


def get_seen_in_costume_count() -> int:
    global seen_in_costume_count
    seen_in_costume_count += 1
    return seen_in_costume_count


def get_seen_neutralizing_count() -> int:
    global seen_neutralizing_count
    seen_neutralizing_count += 1
    return seen_neutralizing_count


def get_seen_killing_target_count() -> int:
    global seen_killing_target_count
    seen_killing_target_count += 1
    return seen_killing_target_count


def heuristic_V2(position, target_position, referee, state):
    x1, y1 = position
    x2, y2 = target_position
    actions_count = get_action_count()
    seen_by_guards_count = 0 if state['is_in_guard_range'] is False else get_seen_by_guards_count(
    )
    seen_in_costume_count = get_seen_in_costume_count() if is_suit_worn(state) and (
        state['is_in_guard_range'] is True or state['is_in_civil_range'] is True) else 0
    seen_neutralizing_count = get_seen_neutralizing_count()
    seen_killing_target_count = get_seen_killing_target_count() if state['is_target_down'] is True and (
        state['is_in_guard_range'] is True or state['is_in_civil_range'] is True) else 0
    heuristic_value = actions_count + (seen_by_guards_count * 5) + (neutralized_count * 20) + \
        (seen_in_costume_count * 100) + (seen_neutralizing_count * 100) + \
        (seen_killing_target_count * 100)
    return heuristic_value


def rotation(referee: HitmanReferee, state: Dict, next_pos: Tuple[int, int]):
    orientation = state['orientation']
    x, y = state['position']

    if (x, y - 1) == next_pos:
        if orientation == HC.W:
            return referee.turn_clockwise()
        elif orientation == HC.E:
            return referee.turn_anti_clockwise()
    elif (x, y + 1) == next_pos:
        if orientation == HC.W:
            return referee.turn_anti_clockwise()
        elif orientation == HC.E:
            return referee.turn_clockwise()
    elif (x - 1, y) == next_pos:
        if orientation == HC.S:
            return referee.turn_clockwise()
        elif orientation == HC.N:
            return referee.turn_anti_clockwise()
    elif (x + 1, y) == next_pos:
        if orientation == HC.S:
            return referee.turn_anti_clockwise()
        elif orientation == HC.N:
            return referee.turn_clockwise()

    return referee.move()


def rotate_matrix(matrix):
    n = len(matrix)
    m = len(matrix[0])
    rotated_matrix = [[None] * n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    return rotated_matrix


def transform_coordinates(x, y, n):
    return y, n - x - 1


def find_object(matrix, obj):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[j][i] == obj:
                return (i, j)
    return None


def is_next_pos_in_vision(status, next_pos):
    vision = status['vision']
    for pos, _ in vision:
        if pos == next_pos:
            return True
    return False

# Obtention des coordonnées des éléments à partir de la grille
coords = grid_to_coords_dict(Grid)

# Initialisation de l'état initial
etat_initial = {}

# Définir les valeurs des prédicats pour les éléments présents dans la grille initiale
for cellule in coords["empty"]:
    etat_initial["at({})".format(cellule)] = True

for hero in coords["hero"]:
    etat_initial["at({})".format(hero)] = True

for rope  in coords["ropes"]:
    etat_initial["at({})".format(rope)] =True

for costume in coords["costumes"]:
    etat_initial["at({})".format(costume)] =True

for garde in coords["guards"]:
    etat_initial["at({})".format(garde)] =True

for target in coords["targets"]:
    etat_initial["at({})".format(target)] =True

# Autres prédicats de l'état initial...






